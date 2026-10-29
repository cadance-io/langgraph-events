"""Tests for the migration gate: ``check``, its render and ``assert_migration_gate``."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from _migration_fixtures import (
    _CLI,
    _DROP_TOTAL,
    _MODULE,
    _OLD_TRIMMED,
    _TRIMMED,
    Asking,
    BadFill,
    Cart,
    Derived,
    FanIn,
    Filled,
    GlobalFill,
    Legacy,
    ReadsDropped,
    Renamed,
    Retiring,
    Shop,
    Splitting,
    SplittingStrict,
    Trimmed,
    TrimmedLambda,
    TrimmedWide,
    Validated,
    ValidatedDefault,
    _add,
    _baselined,
    _changed_key,
    _deleted_transform,
    _entry,
    _ghost,
    _handler_target,
    _key,
    _moved_transform,
    _names,
    _read,
    _target,
)
from conftest import Started

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    IntegrationEvent,
    Namespace,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    AnonymousCallable,
    BaselineStale,
    GateError,
    HandlerMissing,
    Invocation,
    Migration,
    MigrationOutOfScope,
    MigrationTarget,
    SurvivedRelease,
    Uncovered,
    Unrevivable,
    assert_migration_gate,
    check,
    hold_migration,
    update_baseline,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver


class _Problem:
    """A plain mixin that a failure event composes."""

    _event_mixin: ClassVar[bool] = True


@dataclasses.dataclass(frozen=True)
class _Diagnosed:
    """A dataclass mixin that a failure event composes."""

    _event_mixin: ClassVar[bool] = True

    diagnostic: str | None = None


class Faulty(Namespace):
    class Place(Command):
        def handle(self) -> Faulty.Place.Failed:
            return Faulty.Place.Failed()

        class Failed(DomainEvent, _Problem, _Diagnosed):
            pass


@on(_Problem)
def _log_problem(event: _Problem) -> None:
    return None


@on(_Diagnosed)
def _log_diagnosed(event: _Diagnosed) -> None:
    return None


class _Notified(IntegrationEvent):
    note: str = ""


@on(Shop.Place.Placed)
def _notify(event: Shop.Place.Placed) -> _Notified:
    return _Notified(note=event.note)


@on(HandlerRaised)
def _on_raised(event: HandlerRaised) -> None:
    return None


def _scoped_target(
    tmp_path: Path,
    *namespaces: type[Namespace],
    serde_namespaces: tuple[type[Namespace], ...],
    handlers: tuple[Callable[..., object], ...] = (),
) -> MigrationTarget:
    """A target whose serde reads *serde_namespaces* only, and no loose event."""

    def build(saver: BaseCheckpointSaver[Any]) -> EventGraph:
        saver.serde = NamespaceAwareSerde(namespaces=serde_namespaces)
        return EventGraph.from_namespaces(
            *namespaces, handlers=list(handlers) or None, checkpointer=saver
        )

    return MigrationTarget(graph=build, baseline=tmp_path / "baseline.json")


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

        def when_a_recorded_class_transform_follows_a_rename():
            def it_reports_no_deleted_migration(tmp_path: Path):
                (finding,) = check(_moved_transform(tmp_path)).findings

                assert isinstance(finding, BaselineStale)
                assert finding.deleted == ()

        def when_only_the_live_fields_changed():
            def it_reports_the_live_change(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                raw = _read(target.baseline)
                raw["events"][1]["live"] = []
                target.baseline.write_text(json.dumps(raw))

                (finding,) = check(target).findings

                assert finding == BaselineStale(
                    (f"~ event {_MODULE}:Shop.Place.Placed live [] -> [note]",), ()
                )

        def when_a_class_dropped_out_of_the_graph():
            def it_reports_the_retirement(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop))
                _add(target.baseline, "events", {**_ghost(["x"]), "live": ["x"]})

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

    def describe_the_event_scope():
        def when_a_handler_subscribes_to_a_mixin():
            def it_records_the_events_and_no_mixin(tmp_path: Path):
                target = _target(
                    tmp_path, Faulty, handlers=(_log_problem, _log_diagnosed)
                )

                update_baseline(target)

                recorded = {e["qualname"] for e in _read(target.baseline)["events"]}
                assert recorded == {"Faulty.Place", "Faulty.Place.Failed"}

            def it_passes(tmp_path: Path):
                target = _target(
                    tmp_path, Faulty, handlers=(_log_problem, _log_diagnosed)
                )

                assert check(_baselined(target)).findings == ()

        def when_the_serde_reads_no_loose_event():
            def it_records_the_loose_events_of_the_graph(tmp_path: Path):
                target = _scoped_target(
                    tmp_path,
                    Shop,
                    serde_namespaces=(Shop,),
                    handlers=(_notify, _on_raised),
                )

                update_baseline(target)

                recorded = {e["qualname"] for e in _read(target.baseline)["events"]}
                assert {"_Notified", "HandlerRaised"} <= recorded
                assert _read(target.baseline)["retired"] == []

            def it_passes_against_a_baseline_that_records_them(tmp_path: Path):
                handlers = (_notify, _on_raised)
                update_baseline(_target(tmp_path, Shop, handlers=handlers))
                target = _scoped_target(
                    tmp_path, Shop, serde_namespaces=(Shop,), handlers=handlers
                )

                assert check(target).findings == ()

        def when_a_graph_namespace_is_outside_the_serde():
            def it_reports_each_class_whose_migration_the_serde_misses(
                tmp_path: Path,
            ):
                target = _baselined(
                    _scoped_target(tmp_path, Trimmed, Shop, serde_namespaces=(Shop,))
                )

                assert check(target).findings == (
                    MigrationOutOfScope((_MODULE, "Trimmed.Place.Placed")),
                )

            def it_names_the_fix_and_the_next_command(tmp_path: Path):
                target = _baselined(
                    _scoped_target(tmp_path, Trimmed, Shop, serde_namespaces=(Shop,))
                )

                text = check(target).render(Invocation("app.lge:target"))

                assert (
                    "  migration-out-of-scope  "
                    f"{_MODULE}:Trimmed.Place.Placed\n"
                    "    The class declares a read-time migration, but the serde "
                    "does not read the class, so the migration does nothing.\n"
                    "    Pass its namespace in namespaces= of the serde, or the "
                    "class in events=, then run: "
                    f"{_CLI} baseline update app.lge:target"
                ) in text

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
                raw = _read(target.baseline)
                (persisted,) = (
                    e
                    for e in raw["events"]
                    if e["qualname"] == "ReadsDropped.Persisted"
                )
                persisted["fields"] = ["legacy_flag", "note"]
                target.baseline.write_text(json.dumps(raw))

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert finding.identity == (_MODULE, "ReadsDropped.Persisted")
                assert "TransformFields raised AttributeError" in finding.error
                assert "None placeholder" in finding.error

        def when_a_select_returns_None_on_the_placeholder():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(Splitting,)))

                assert check(target).findings == ()

        def when_a_select_reads_the_placeholder():
            def it_names_the_placeholder_contract(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(SplittingStrict,)))

                (finding,) = check(target).findings

                assert isinstance(finding, Unrevivable)
                assert "SplitEvent raised TypeError" in finding.error

        def when_an_origin_fill_supplies_a_validated_field():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(FanIn,)))
                _record_retired(target, FanIn, "FanInOld.Approved", [])

                findings = check(target).findings

                assert [f for f in findings if "FanInOld" in f.subject] == []

        def when_a_class_fill_supplies_a_validated_field():
            def it_passes(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(GlobalFill,)))

                assert check(target).findings == ()

        def when_a_fill_value_fails_validation():
            def it_reports_the_identity(tmp_path: Path):
                target = _baselined(_target(tmp_path, Shop, scope=(BadFill,)))

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

            def when_the_code_dropped_a_field_the_previous_release_wrote():
                def it_reports_the_field_after_the_prune_update(tmp_path: Path):
                    target, previous = _deleted_transform(tmp_path, TrimmedWide)
                    update_baseline(target, previous=previous)

                    (finding,) = check(target, previous=previous).findings

                    assert isinstance(finding, Unrevivable)
                    assert "The baseline recorded 'legacy_b'." in finding.error

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

            def when_the_next_release_holds_the_same_migrations():
                def it_expires(tmp_path: Path):
                    target = _target(tmp_path, Trimmed)
                    previous = _previous(tmp_path, target)
                    hold_migration(
                        target,
                        _TRIMMED,
                        reason="a subgraph needs it",
                        previous=previous,
                    )
                    next_release = tmp_path / "next.json"
                    next_release.write_text(target.baseline.read_text())

                    (finding,) = check(target, previous=next_release).findings

                    assert isinstance(finding, SurvivedRelease)
                    assert finding.expired is not None

            def when_the_previous_file_is_formatted_differently():
                def it_still_applies(tmp_path: Path):
                    target = _target(tmp_path, Trimmed)
                    previous = _previous(tmp_path, target)
                    hold_migration(
                        target,
                        _TRIMMED,
                        reason="a subgraph needs it",
                        previous=previous,
                    )
                    previous.write_text(json.dumps(_read(previous)))

                    assert check(target, previous=previous).findings == ()

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
                    "app.lge:target --previous ci/prev.json\n"
                    f"    Or keep it one more release: {_CLI} baseline hold "
                    f"app.lge:target '{_TRIMMED}' --reason <why> --previous "
                    "ci/prev.json"
                )

        def when_a_migration_key_changed():
            def it_names_the_new_key_of_the_same_kind_and_identity(tmp_path: Path):
                target, previous = _changed_key(tmp_path)

                text = check(target, previous=previous).render(
                    Invocation("app.lge:target", "prev.json")
                )

                assert (
                    f"    {_OLD_TRIMMED} and {_TRIMMED} are of the same kind on "
                    "the same event, so the key of one migration can have "
                    "changed. The history of an event that keeps a migration "
                    "stays.\n"
                ) in text

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
                    f"  held  {label}: subgraph (made against lge-b1-"
                )

        def when_a_hold_expired():
            def it_offers_a_new_hold_and_does_not_list_it_as_held(tmp_path: Path):
                target = _target(tmp_path, Trimmed)
                update_baseline(target)
                previous = tmp_path / "prev.json"
                previous.write_text(target.baseline.read_text())
                hold_migration(target, _TRIMMED, reason="subgraph", previous=previous)
                next_release = tmp_path / "next.json"
                next_release.write_text(target.baseline.read_text())

                report = check(target, previous=next_release)
                lines = report.render(Invocation("app.lge:target", "next.json"))

                assert report.held == ()
                assert lines.splitlines()[-2:] == [
                    f"    Delete it, then run: {_CLI} baseline update app.lge:target "
                    "--previous next.json",
                    f"    Or keep it one more release: {_CLI} baseline hold "
                    f"app.lge:target '{_TRIMMED}' --reason <why> --previous next.json",
                ]


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
