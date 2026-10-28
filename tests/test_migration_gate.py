"""Tests for the migration gate: the baseline, ``check`` and the CLI."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from conftest import Started

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Interrupted,
    Namespace,
    on,
)
from langgraph_events.agui import FrontendToolCallRequested
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    AnonymousCallable,
    BaselineStale,
    GateError,
    HandlerMissing,
    Invocation,
    Migration,
    MigrationKey,
    MigrationTarget,
    SurvivedRelease,
    TargetError,
    Uncovered,
    Unrevivable,
    assert_migration_gate,
    backfill,
    check,
    hold_migration,
    migrate_from,
    split_event,
    transform_fields,
    update_baseline,
)
from langgraph_events.serde.migrations.__main__ import main

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver

_MODULE = __name__
_CLI = "python -m langgraph_events.serde.migrations"
_DROP_TOTAL = f"{_MODULE}:_drop_total"
_TRIMMED = f"transform {_MODULE}:Trimmed.Place.Placed via {_DROP_TOTAL}"


def _drop_total(kw: dict[str, Any]) -> dict[str, Any]:
    kw.pop("total", None)
    return kw


def _upper_legacy_flag(kw: dict[str, Any]) -> dict[str, Any]:
    """Raises ``AttributeError`` on the gate's ``None`` placeholder."""
    kw["note"] = kw.pop("legacy_flag").upper()
    return kw


def _select_none(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None
    return Splitting.Failed, {"reason": result["message"]}


def _select_reading_placeholder(kw: dict[str, Any]) -> tuple[type, dict[str, Any]]:
    """Raises ``TypeError`` on the gate's ``None`` placeholder."""
    return SplittingStrict.Failed, {"reason": kw["result"]["message"]}


class Shop(Namespace):
    class Place(Command):
        note: str = ""

        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> Shop.Place.Placed:
            return Shop.Place.Placed(note=self.note)


class Renamed(Namespace):
    class Place(Command):
        note: str = ""

        @migrate_from("Renamed.Placed")
        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> Renamed.Place.Placed:
            return Renamed.Place.Placed(note=self.note)


class Trimmed(Namespace):
    class Place(Command):
        note: str = ""

        @transform_fields(_drop_total)
        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> Trimmed.Place.Placed:
            return Trimmed.Place.Placed(note=self.note)


class TrimmedLambda(Namespace):
    class Place(Command):
        note: str = ""

        @transform_fields(lambda kw: kw)
        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> TrimmedLambda.Place.Placed:
            return TrimmedLambda.Place.Placed(note=self.note)


class Filled(Namespace):
    class Place(Command):
        note: str = ""

        @backfill("status", default="open")
        class Placed(DomainEvent):
            status: str
            note: str = ""

        def handle(self) -> Filled.Place.Placed:
            return Filled.Place.Placed(status="open", note=self.note)


class Validated(Namespace):
    class Place(Command):
        name: str = "x"

        class Placed(DomainEvent):
            name: str

            def __post_init__(self) -> None:
                if not self.name:
                    raise ValueError("name must be non-empty")

        def handle(self) -> Validated.Place.Placed:
            return Validated.Place.Placed(name=self.name)


class ValidatedDefault(Namespace):
    class Place(Command):
        name: str = "x"

        class Placed(DomainEvent):
            name: str = "x"

            def __post_init__(self) -> None:
                if not self.name:
                    raise ValueError("name must be non-empty")

        def handle(self) -> ValidatedDefault.Place.Placed:
            return ValidatedDefault.Place.Placed(name=self.name)


class Derived(Namespace):
    class Place(Command):
        note: str = ""

        class Placed(DomainEvent):
            note: str = ""
            length: int = dataclasses.field(init=False)

            def __post_init__(self) -> None:
                object.__setattr__(self, "length", len(self.note))

        def handle(self) -> Derived.Place.Placed:
            return Derived.Place.Placed(note=self.note)


class Asking(Namespace):
    class Ask(Command):
        name: str = "confirm"

        def handle(self) -> FrontendToolCallRequested:
            return FrontendToolCallRequested(name=self.name)


# Tombstones for identities a baseline lists under ``retired``.
class Retiring(Namespace):
    @migrate_from("Retiring.ApprovalRequired")
    class RetiredApprovalGate(Interrupted):
        order_id: str = ""

    @migrate_from("Retiring.ReviewRequired")
    class RetiredReviewGate(Interrupted):
        order_id: str


class ReadsDropped(Namespace):
    @transform_fields(_upper_legacy_flag)
    class Persisted(DomainEvent):
        note: str = ""


class FanIn(Namespace):
    @migrate_from("FanInOld.Approved", backfill={"entity_type": "persona"})
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


class GlobalFill(Namespace):
    @backfill("entity_type", default="persona")
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


class BadFill(Namespace):
    @backfill("entity_type", default="bogus")
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


class Splitting(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(_select_none, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any]


class SplittingStrict(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(_select_reading_placeholder, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any]


class Cart(Namespace):
    class Checkout(Command):
        note: str = ""

        class CheckedOut(DomainEvent):
            note: str = ""

        def handle(self) -> Cart.Checkout.CheckedOut:
            return Cart.Checkout.CheckedOut(note=self.note)


class Legacy(Namespace):
    class Place(Command):
        note: str = ""

        @migrate_from("Legacy.OldPlaced", backfill={"source": "legacy"})
        class Placed(DomainEvent):
            source: str
            note: str = ""

        def handle(self) -> Legacy.Place.Placed:
            return Legacy.Place.Placed(source="live", note=self.note)


def _target(
    tmp_path: Path,
    *namespaces: type[Namespace],
    scope: Sequence[type] = (),
    migrations: Sequence[Migration] = (),
    handlers: Sequence[Callable[..., Any]] = (),
    name: str = "baseline.json",
) -> MigrationTarget:
    """A target over *namespaces*. *scope* adds namespaces the serde reads."""

    def build(saver: BaseCheckpointSaver[Any]) -> EventGraph:
        if scope or migrations:
            saver.serde = NamespaceAwareSerde(
                migrations, namespaces=(*namespaces, *scope)
            )
        return EventGraph.from_namespaces(
            *namespaces, handlers=list(handlers) or None, checkpointer=saver
        )

    return MigrationTarget(graph=build, baseline=tmp_path / name)


def _handler_target(
    tmp_path: Path, *handlers: Callable[..., Any] | type
) -> MigrationTarget:
    def build(saver: BaseCheckpointSaver[Any]) -> EventGraph:
        return EventGraph.from_namespaces(handlers=list(handlers), checkpointer=saver)

    return MigrationTarget(graph=build, baseline=tmp_path / "baseline.json")


def _read(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def _add(path: Path, key: str, *entries: dict[str, Any]) -> None:
    """Append *entries* to the list *key* of the baseline at *path*."""
    raw = _read(path)
    raw.setdefault(key, []).extend(entries)
    path.write_text(json.dumps(raw))


def _entry(module: str, qualname: str, fields: list[str] | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"module": module, "qualname": qualname}
    if fields is not None:
        entry["fields"] = fields
    return entry


def _ghost(fields: list[str] | None = None) -> dict[str, Any]:
    return _entry("ghost.mod", "Ghost.Gone", fields)


def _fields(path: Path, list_name: str = "events") -> dict[str, list[str] | None]:
    return {e["qualname"]: e.get("fields") for e in _read(path)[list_name]}


def _names(findings: Sequence[object]) -> list[str]:
    return [type(finding).__name__ for finding in findings]


def _key(label: str) -> MigrationKey:
    return MigrationKey.parse(label)


def _deleted_transform(tmp_path: Path) -> tuple[MigrationTarget, Path]:
    """Release N held a transform on ``Shop.Place.Placed`` that dropped
    ``total``. The code of release N+1 deleted it. Returns the N+1 target,
    whose committed baseline is still N's, and N's baseline."""
    shipped = _target(tmp_path, Trimmed, name="previous.json")
    update_baseline(shipped)
    raw = json.loads(shipped.baseline.read_text().replace("Trimmed", "Shop"))
    raw["events"][1]["fields"] = ["note", "total"]
    shipped.baseline.write_text(json.dumps(raw))
    target = _target(tmp_path, Shop)
    target.baseline.write_text(shipped.baseline.read_text())
    return target, shipped.baseline


def describe_update_baseline():
    def when_no_baseline_exists_yet():
        def it_writes_a_version_4_file(tmp_path: Path):
            target = _target(tmp_path, Renamed)

            update_baseline(target)

            assert _read(target.baseline) == {
                "version": 4,
                "events": [
                    _entry(_MODULE, "Renamed.Place", ["note"]),
                    _entry(_MODULE, "Renamed.Place.Placed", ["note"]),
                ],
                "retired": [],
                "handlers": [{"name": "Renamed.Place"}],
                "migrations": [{"label": f"rename {_MODULE}:Renamed.Placed"}],
            }

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
            assert placed in _read(target.baseline)["events"]

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

    def when_a_v4_migration_TRIMMEDis_malformed():
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
            def it_keeps_the_history(tmp_path: Path):
                target, _ = _deleted_transform(tmp_path)

                update_baseline(target)

                assert _fields(target.baseline)["Shop.Place.Placed"] == [
                    "note",
                    "total",
                ]

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


def _baselined(target: MigrationTarget) -> MigrationTarget:
    update_baseline(target)
    return target


def describe_check():
    def when_the_baseline_matches_the_code():
        def it_passes_every_rule_but_release(tmp_path: Path):
            report = check(_baselined(_target(tmp_path, Renamed)))

            assert report.findings == ()
            assert report.rules_run == {
                "baseline",
                "names",
                "coverage",
                "revive",
                "handlers",
            }
            assert report.release_not_run == "no previous baseline was given"

    def when_the_label_order_differs_from_the_key_order():
        def it_passes_after_an_update(tmp_path: Path):
            renames = (
                Migration.rename(
                    old_module="legacy.x", old_qualname="A", to=Shop.Place.Placed
                ),
                Migration.rename(old_module="legacy", old_qualname="B", to=Shop.Place),
            )

            report = check(_baselined(_target(tmp_path, Shop, migrations=renames)))

            assert report.findings == ()

    def describe_the_baseline_rule():
        def when_the_code_added_a_migration():
            def it_reports_the_new_key(tmp_path: Path):
                target = _baselined(_target(tmp_path, Renamed))
                raw = _read(target.baseline)
                raw["migrations"] = []
                target.baseline.write_text(json.dumps(raw))

                (finding,) = check(target).findings

                assert finding == BaselineStale(
                    (f"+ migration rename {_MODULE}:Renamed.Placed",), ()
                )

        def when_no_baseline_exists_yet():
            def it_reports_every_line_to_write(tmp_path: Path):
                (finding,) = check(_target(tmp_path, Shop)).findings

                assert finding == BaselineStale(
                    (
                        f"+ event {_MODULE}:Shop.Place",
                        f"+ event {_MODULE}:Shop.Place.Placed",
                        "+ handler Shop.Place",
                    ),
                    (),
                )

        def when_the_baseline_predates_v4():
            def it_reports_the_version(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                raw = _read(target.baseline)
                raw["version"] = 3
                del raw["migrations"]
                target.baseline.write_text(json.dumps(raw))

                (finding,) = check(target).findings

                assert finding == BaselineStale(("~ version 3 -> 4",), ())

        def when_a_class_dropped_out_of_the_graph():
            def it_reports_the_retirement(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                _add(target.baseline, "events", _ghost(["x"]))

                findings = check(target).findings

                assert findings[0] == BaselineStale(
                    ("- event ghost.mod:Ghost.Gone", "+ retired ghost.mod:Ghost.Gone"),
                    (),
                )

    def describe_the_names_rule():
        def when_a_transform_is_a_lambda():
            def it_asks_for_a_module_level_name(tmp_path: Path):
                target = _baselined(_target(tmp_path, TrimmedLambda))

                (finding,) = check(target).findings

                assert finding == AnonymousCallable(
                    _key(
                        f"transform {_MODULE}:TrimmedLambda.Place.Placed "
                        f"via {_MODULE}:TrimmedLambda.Place.<lambda>"
                    )
                )

    def describe_the_coverage_rule():
        def when_a_retired_identity_has_no_migration():
            def it_reports_the_identity(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                _add(target.baseline, "retired", _ghost(["x"]))

                assert check(target).findings == (
                    Uncovered(("ghost.mod", "Ghost.Gone"), retired=True),
                )

        def when_a_live_class_shares_the_last_name():
            def it_names_the_rename_candidate(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                _add(target.baseline, "retired", _entry("old.mod", "Old.Placed", []))

                (finding,) = check(target).findings

                assert finding == Uncovered(
                    ("old.mod", "Old.Placed"),
                    retired=True,
                    renamed_to=((_MODULE, "Shop.Place.Placed"),),
                )

        def when_several_live_classes_share_the_last_name():
            def it_names_every_candidate(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, Legacy))
                _add(target.baseline, "retired", _entry("old.mod", "Old.Placed", []))

                (finding,) = check(target).findings

                assert isinstance(finding, Uncovered)
                assert finding.renamed_to == (
                    (_MODULE, "Legacy.Place.Placed"),
                    (_MODULE, "Shop.Place.Placed"),
                )

        def when_a_rename_covers_the_retired_identity():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Renamed))
                _add(
                    target.baseline,
                    "retired",
                    _entry(_MODULE, "Renamed.Placed", ["note"]),
                )

                assert check(target).findings == ()

        def when_a_hand_written_migration_covers_it():
            def it_passes(tmp_path: Path):
                migration = Migration.rename(
                    old_module="legacy.mod",
                    old_qualname="Old.Placed",
                    to=Shop.Place.Placed,
                )
                target = _baselined(_target(tmp_path, Shop, migrations=(migration,)))
                _add(target.baseline, "retired", _entry("legacy.mod", "Old.Placed", []))

                assert check(target).findings == ()

    def describe_the_revive_rule():
        def _record_retired(
            target: MigrationTarget,
            namespace: type,
            name: str,
            fields: list[str] | None,
        ) -> MigrationTarget:
            _add(target.baseline, "retired", _entry(namespace.__module__, name, fields))
            return target

        def when_the_live_class_dropped_a_recorded_field():
            def it_names_the_recorded_field(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                raw = _read(target.baseline)
                raw["events"][1]["fields"] = ["legacy_flag", "note"]
                target.baseline.write_text(json.dumps(raw))

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert finding.identity == (_MODULE, "Shop.Place.Placed")
                assert "The baseline recorded 'legacy_flag'." in finding.error
                assert "@transform_fields" in finding.error
                assert finding.error.count("from the payload") == 1

        def when_a_transform_drops_the_recorded_field():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Trimmed))
                raw = _read(target.baseline)
                raw["events"][1]["fields"] = ["note", "total"]
                target.baseline.write_text(json.dumps(raw))

                assert check(target).findings == ()

        def when_a_required_field_has_no_fill():
            def it_sends_a_placeholder_for_it(tmp_path: Path):
                assert check(_baselined(_target(tmp_path, Legacy))).findings == ()

        def when_a_class_is_reachable_only_in_scope():
            def it_revives_through_the_scope(tmp_path: Path):
                class Trading(Namespace):
                    class Place(Command):
                        sym: str

                        class Placed(DomainEvent):
                            sym: str

                        def handle(self) -> Any:
                            return Trading.Place.Placed(sym=self.sym)

                # A local class has no importable name, so the string
                # annotation cannot resolve: give the class itself.
                Trading.Place.handle.__annotations__["return"] = Trading.Place.Placed
                target = _baselined(_target(tmp_path, Trading))

                assert check(target).findings == ()

        def when_a_required_field_has_a_backfill():
            def it_lets_the_fill_supply_it(tmp_path: Path):
                target = _baselined(_target(tmp_path, Filled))
                raw = _read(target.baseline)
                raw["events"][1]["fields"] = ["note"]
                target.baseline.write_text(json.dumps(raw))

                assert _names(check(target).findings) == ["BaselineStale"]

        def when_the_live_class_has_an_init_false_field():
            def it_passes(tmp_path: Path):
                assert check(_baselined(_target(tmp_path, Derived))).findings == ()

        def when_a_defaulted_field_is_validated():
            def it_sends_no_placeholder_for_it(tmp_path: Path):
                target = _baselined(_target(tmp_path, ValidatedDefault))

                assert check(target).findings == ()

        def when_a_required_field_is_validated():
            def it_reports_the_placeholder_failure(tmp_path: Path):
                target = _baselined(_target(tmp_path, Validated))

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert finding.identity == (_MODULE, "Validated.Place.Placed")

        def when_a_library_event_validates_a_required_field():
            def it_only_resolves_the_identity(tmp_path: Path):
                target = _baselined(_target(tmp_path, Asking))

                assert check(target).findings == ()

        def when_a_tombstone_declares_the_recorded_fields():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(Retiring,)))
                _record_retired(
                    target, Retiring, "Retiring.ApprovalRequired", ["order_id"]
                )

                assert check(target).findings == ()

        def when_a_retired_record_predates_v3():
            def it_sends_required_placeholders_only(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(Retiring,)))
                _record_retired(target, Retiring, "Retiring.ReviewRequired", None)

                assert check(target).findings == ()

        def when_a_transform_reads_a_dropped_field():
            def it_names_the_placeholder_contract(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(ReadsDropped,)))
                _record_retired(
                    target,
                    ReadsDropped,
                    "ReadsDropped.Persisted",
                    ["legacy_flag", "note"],
                )

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert finding.retired
                assert "TransformFields raised AttributeError" in finding.error
                assert "None placeholder" in finding.error

        def when_a_select_returns_None_on_the_placeholder():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(Splitting,)))
                _record_retired(target, Splitting, "Splitting.Completed", ["result"])

                assert check(target).findings == ()

        def when_a_select_reads_the_placeholder():
            def it_names_the_placeholder_contract(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(SplittingStrict,)))
                _record_retired(
                    target, SplittingStrict, "SplittingStrict.Completed", ["result"]
                )

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert "SplitEvent raised TypeError" in finding.error

        def when_an_origin_fill_supplies_a_validated_field():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(FanIn,)))
                _record_retired(target, FanIn, "FanInOld.Approved", [])

                assert check(target).findings == ()

        def when_a_class_fill_supplies_a_validated_field():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(GlobalFill,)))
                _record_retired(target, GlobalFill, "GlobalFill.Approved", [])

                assert check(target).findings == ()

        def when_a_fill_value_fails_validation():
            def it_reports_the_identity(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(BadFill,)))
                _record_retired(target, BadFill, "BadFill.Approved", [])

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert finding.identity == (_MODULE, "BadFill.Approved")

    def describe_the_handlers_rule():
        def when_every_recorded_handler_is_live():
            def it_passes(tmp_path: Path):
                @on(Started)
                def place(event: Started) -> None:
                    return None

                assert (
                    check(_baselined(_handler_target(tmp_path, place))).findings == ()
                )

        def when_a_handler_is_renamed():
            def with_previously_declared():
                def it_reports_only_the_stale_baseline(tmp_path: Path):
                    @on(Started)
                    def place(event: Started) -> None:
                        return None

                    @on(Started, previously="place")
                    def submit(event: Started) -> None:
                        return None

                    _baselined(_handler_target(tmp_path, place))

                    findings = check(_handler_target(tmp_path, submit)).findings

                    assert _names(findings) == ["BaselineStale"]

            def without_previously_declared():
                def it_reports_the_lost_handler(tmp_path: Path):
                    @on(Started)
                    def place(event: Started) -> None:
                        return None

                    @on(Started)
                    def submit(event: Started) -> None:
                        return None

                    _baselined(_handler_target(tmp_path, place))

                    findings = check(_handler_target(tmp_path, submit)).findings

                    assert findings[1:] == (HandlerMissing("place"),)

        def when_a_reactor_is_replaced_by_an_inline_command():
            def with_previously_declared():
                def it_keeps_the_handler_covered(tmp_path: Path):
                    class _Replat(Namespace):
                        class Persist(Command):
                            previously: ClassVar = ("persist",)

                            def handle(self) -> None:
                                return None

                    @on(Started, node_name="persist")
                    def persist_reactor(event: Started) -> None:
                        return None

                    _baselined(_handler_target(tmp_path, persist_reactor))

                    findings = check(
                        _handler_target(tmp_path, _Replat.Persist)
                    ).findings

                    assert HandlerMissing("persist") not in findings

            def without_previously_declared():
                def it_reports_the_lost_handler(tmp_path: Path):
                    class _Replat2(Namespace):
                        class Persist(Command):
                            def handle(self) -> None:
                                return None

                    @on(Started, node_name="persist")
                    def persist_reactor(event: Started) -> None:
                        return None

                    _baselined(_handler_target(tmp_path, persist_reactor))

                    findings = check(
                        _handler_target(tmp_path, _Replat2.Persist)
                    ).findings

                    assert HandlerMissing("persist") in findings

        def when_the_baseline_recorded_old_positional_names():
            def it_passes_after_one_update(tmp_path: Path):
                target = _handler_target(tmp_path, Shop.Place, Cart.Checkout)
                target.baseline.write_text(
                    json.dumps(
                        {
                            "version": 2,
                            "events": [],
                            "handlers": [{"name": "handle"}, {"name": "handle_2"}],
                        }
                    )
                )
                assert check(target).findings[1:] == (
                    HandlerMissing("handle"),
                    HandlerMissing("handle_2"),
                )

                update_baseline(target)

                assert check(target).findings == ()

        def when_the_baseline_predates_handler_tracking():
            def it_reports_only_the_stale_baseline(tmp_path: Path):
                @on(Started)
                def place(event: Started) -> None:
                    return None

                target = _handler_target(tmp_path, place)
                target.baseline.write_text(json.dumps({"version": 1, "events": []}))

                assert _names(check(target).findings) == ["BaselineStale"]

    def describe_the_release_rule():
        def _previous(tmp_path: Path, target: MigrationTarget) -> Path:
            """Commit the baseline of *target* as the previous release's."""
            update_baseline(target)
            previous = tmp_path / "previous.json"
            previous.write_text(target.baseline.read_text())
            return previous

        def when_a_shipped_migration_is_still_in_the_code():
            def it_reports_the_survivor(tmp_path: Path):
                target = _target(tmp_path, Trimmed)
                previous = _previous(tmp_path, target)

                report = check(target, previous=previous)

                assert report.findings == (
                    SurvivedRelease(
                        _key(
                            f"transform {_MODULE}:Trimmed.Place.Placed "
                            f"via {_MODULE}:_drop_total"
                        )
                    ),
                )
                assert "release" in report.rules_run

        def when_a_shipped_class_transform_follows_a_rename():
            def it_reports_the_moved_key(tmp_path: Path):
                target = _target(tmp_path, Trimmed)
                previous = _previous(tmp_path, target)
                raw = _read(previous)
                raw["migrations"] = [
                    {"label": f"transform {_MODULE}:Old.Placed via {_DROP_TOTAL}"}
                ]
                previous.write_text(json.dumps(raw))
                moved = _target(
                    tmp_path,
                    Trimmed,
                    migrations=(
                        Migration.rename(
                            old_module=_MODULE,
                            old_qualname="Old.Placed",
                            to=Trimmed.Place.Placed,
                        ),
                    ),
                )
                update_baseline(moved)

                findings = check(moved, previous=previous).findings

                assert _names(findings) == ["SurvivedRelease"]

        def when_the_shipped_callable_is_anonymous():
            def it_reports_only_the_name(tmp_path: Path):
                target = _target(tmp_path, TrimmedLambda)
                previous = _previous(tmp_path, target)

                findings = check(target, previous=previous).findings

                assert _names(findings) == ["AnonymousCallable"]

        def when_a_shipped_migration_was_deleted():
            def it_asks_for_the_prune_update_only(tmp_path: Path):
                target, previous = _deleted_transform(tmp_path)

                (finding,) = check(target, previous=previous).findings

                assert isinstance(finding, BaselineStale)
                assert finding.deleted == (
                    _key(f"transform {_MODULE}:Shop.Place.Placed via {_DROP_TOTAL}"),
                )

            def without_previous():
                def it_asks_for_the_prune_update_only(tmp_path: Path):
                    target, _ = _deleted_transform(tmp_path)

                    findings = check(target).findings

                    assert _names(findings) == ["BaselineStale"]

        def when_the_previous_baseline_predates_v4():
            def it_says_the_rule_did_not_run(tmp_path: Path):
                target = _baselined(_target(tmp_path, Trimmed))
                previous = tmp_path / "previous.json"
                previous.write_text(json.dumps({"version": 3, "events": []}))

                report = check(target, previous=previous)

                assert report.findings == ()
                assert "release" not in report.rules_run
                assert (
                    report.release_not_run
                    == "the previous baseline records no migrations"
                )

        def when_the_previous_baseline_is_missing():
            def it_raises_naming_the_path(tmp_path: Path):
                target = _baselined(_target(tmp_path, Trimmed))

                with pytest.raises(ValueError, match="No previous baseline at"):
                    check(target, previous=tmp_path / "absent.json")

        def describe_a_hold():
            def when_it_was_made_against_this_previous_baseline():
                def it_suppresses_the_survivor(tmp_path: Path):
                    target = _target(tmp_path, Trimmed)
                    previous = _previous(tmp_path, target)
                    hold_migration(
                        target,
                        _TRIMMED,
                        reason="a subgraph needs it",
                        previous=previous,
                    )

                    report = check(target, previous=previous)

                    assert report.findings == ()
                    assert [r.key.label for r in report.held] == [_TRIMMED]

            def when_it_was_made_against_an_older_release():
                def it_expires(tmp_path: Path):
                    target = _target(tmp_path, Trimmed)
                    previous = _previous(tmp_path, target)
                    hold_migration(
                        target,
                        _TRIMMED,
                        reason="a subgraph needs it",
                        previous=previous,
                    )
                    _add(previous, "migrations", {"label": f"rename {_MODULE}:Gone"})

                    (finding,) = check(target, previous=previous).findings

                    assert isinstance(finding, SurvivedRelease)
                    assert finding.expired is not None
                    assert finding.expired.reason == "a subgraph needs it"

    def describe_render():
        def when_the_gate_passes():
            def it_states_the_rules_it_ran(tmp_path: Path):
                report = check(_baselined(_target(tmp_path, Shop)))

                assert report.render(Invocation("app.lge:target")) == (
                    "lge migration gate: passed (rules run: baseline, names, "
                    "coverage, revive, handlers; release rule not run: no "
                    "previous baseline was given)"
                )

        def when_a_shipped_migration_survived():
            def it_prints_the_callers_arguments_in_the_command(tmp_path: Path):
                target = _target(tmp_path, Trimmed)
                update_baseline(target)
                previous = tmp_path / "prev.json"
                previous.write_text(target.baseline.read_text())

                text = check(target, previous=previous).render(
                    Invocation("app.lge:target", "ci/prev.json")
                )

                assert text == (
                    "lge migration gate: 1 finding (rules run: baseline, names, "
                    "coverage, revive, handlers, release)\n"
                    f"  survived-release  transform {_MODULE}:Trimmed.Place.Placed "
                    f"via {_MODULE}:_drop_total\n"
                    "    The previous release shipped this migration.\n"
                    f"    Delete it, then run: {_CLI} baseline update "
                    "app.lge:target --previous ci/prev.json"
                )

        def when_a_deletion_needs_a_previous_baseline():
            def it_prints_a_placeholder_for_it(tmp_path: Path):
                target, _ = _deleted_transform(tmp_path)

                text = check(target).render(Invocation("app.lge:target"))

                assert text.endswith(
                    f"Run: {_CLI} baseline update app.lge:target "
                    "--previous <baseline file of the previous release>"
                )

        def when_a_hold_applies():
            def it_lists_the_hold(tmp_path: Path):
                target = _target(tmp_path, Trimmed)
                update_baseline(target)
                previous = tmp_path / "prev.json"
                previous.write_text(target.baseline.read_text())
                label = _TRIMMED
                hold_migration(target, label, reason="subgraph", previous=previous)

                text = check(target, previous=previous).render(
                    Invocation("app.lge:target")
                )

                assert text.splitlines()[1].startswith(
                    f"  held  {label}: subgraph (made against lge-m1-"
                )


def describe_assert_migration_gate():
    def when_the_gate_passes():
        def it_returns(tmp_path: Path):
            assert_migration_gate(_baselined(_target(tmp_path, Shop)))

    def when_the_gate_finds_a_problem():
        def it_raises_the_rendered_report(tmp_path: Path):
            target = _baselined(_target(tmp_path, Shop))
            _add(target.baseline, "retired", _ghost(["x"]))

            with pytest.raises(GateError) as excinfo:
                assert_migration_gate(target)

            assert excinfo.value.report.findings == (
                Uncovered(("ghost.mod", "Ghost.Gone"), retired=True),
            )
            assert isinstance(excinfo.value, AssertionError)
            assert (
                f"then run: {_CLI} baseline update <module:attr of the MigrationTarget>"
                in str(excinfo.value)
            )


def describe_hold_migration():
    def when_the_previous_release_shipped_the_migration():
        def it_records_the_hold_against_the_previous_fingerprint(tmp_path: Path):
            target = _baselined(_target(tmp_path, Trimmed))
            previous = tmp_path / "prev.json"
            previous.write_text(target.baseline.read_text())

            hold_migration(target, _TRIMMED, reason="subgraph", previous=previous)

            fingerprint = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    "-m",
                    "langgraph_events.serde.migrations",
                    "fingerprint",
                    str(previous),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            assert _read(target.baseline)["migrations"] == [
                {
                    "label": _TRIMMED,
                    "hold": {"reason": "subgraph", "against": fingerprint},
                }
            ]

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


CLI_TARGET: MigrationTarget | None = None


def describe_cli():
    def _install(monkeypatch: pytest.MonkeyPatch, target: MigrationTarget) -> str:
        monkeypatch.setattr(sys.modules[__name__], "CLI_TARGET", target)
        return f"{_MODULE}:CLI_TARGET"

    def describe_check():
        def when_the_gate_passes():
            def it_exits_zero(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _baselined(_target(tmp_path, Shop)))

                assert main(["check", spec]) == 0
                assert capsys.readouterr().out.startswith("lge migration gate: passed")

        def when_the_gate_finds_a_problem():
            def it_exits_one_and_prints_the_callers_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _target(tmp_path, Shop))

                assert main(["check", spec]) == 1
                assert f"Run: {_CLI} baseline update {spec}" in capsys.readouterr().out

        def when_the_target_does_not_import():
            def it_exits_two(capsys: pytest.CaptureFixture[str]):
                assert main(["check", "no_such_module:target"]) == 2
                assert "does not import" in capsys.readouterr().err

        def when_the_attribute_is_not_a_target():
            def it_exits_two(capsys: pytest.CaptureFixture[str]):
                assert main(["check", f"{_MODULE}:_target"]) == 2
                assert "not a MigrationTarget" in capsys.readouterr().err

    def describe_baseline_update():
        def when_the_target_imports():
            def it_writes_the_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                target = _target(tmp_path, Shop)
                spec = _install(monkeypatch, target)

                assert main(["baseline", "update", spec]) == 0
                assert _read(target.baseline)["version"] == 4

        def when_the_previous_file_is_missing():
            def it_exits_two(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _target(tmp_path, Shop))

                code = main(
                    ["baseline", "update", spec, "--previous", str(tmp_path / "x.json")]
                )

                assert code == 2
                assert "No previous baseline at" in capsys.readouterr().err

    def describe_baseline_hold():
        def when_every_argument_is_given():
            def it_writes_the_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                target = _baselined(_target(tmp_path, Trimmed))
                previous = tmp_path / "prev.json"
                previous.write_text(target.baseline.read_text())
                label = _TRIMMED
                spec = _install(monkeypatch, target)

                code = main(
                    [
                        "baseline",
                        "hold",
                        spec,
                        label,
                        "--reason",
                        "subgraph",
                        "--previous",
                        str(previous),
                    ]
                )

                assert code == 0
                assert (
                    _read(target.baseline)["migrations"][0]["hold"]["reason"]
                    == "subgraph"
                )

        def when_the_reason_is_missing():
            def it_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                spec = _install(monkeypatch, _target(tmp_path, Trimmed))

                assert main(["baseline", "hold", spec, "rename m:X"]) == 2

    def describe_fingerprint():
        def when_the_file_records_migrations():
            def it_prints_the_fingerprint_of_the_file(
                tmp_path: Path, capsys: pytest.CaptureFixture[str]
            ):
                baseline = tmp_path / "baseline.json"
                baseline.write_text(
                    json.dumps(
                        {
                            "version": 4,
                            "events": [],
                            "migrations": [{"label": "rename m:Old"}],
                        }
                    )
                )

                assert main(["fingerprint", str(baseline)]) == 0
                assert capsys.readouterr().out == "lge-m1-dbc7afada42b30d1\n"

        def when_the_file_predates_v4():
            def it_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
                baseline = tmp_path / "baseline.json"
                baseline.write_text(json.dumps({"version": 3, "events": []}))

                assert main(["fingerprint", str(baseline)]) == 2
                assert "records no migrations" in capsys.readouterr().err

    def when_the_removed_positional_form_is_used():
        def it_exits_two_and_prints_the_new_command(
            capsys: pytest.CaptureFixture[str],
        ):
            assert main(["myapp.graph:build", "baseline.json"]) == 2
            assert (
                f"{_CLI} check <module:attr of the MigrationTarget>"
                in capsys.readouterr().err
            )

    def when_invoked_as_a_module():
        def it_advertises_the_working_entrypoint():
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "langgraph_events.serde.migrations"],
                capture_output=True,
                check=False,
            )

            assert result.returncode == 2
            assert (
                b"usage: python -m langgraph_events.serde.migrations" in result.stderr
            )
