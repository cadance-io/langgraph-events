"""Tests for the natural key of each client read-time migration."""

from __future__ import annotations

import functools
from typing import Any

import pytest

from langgraph_events import DomainEvent, Namespace
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    AddField,
    Migration,
    MigrationKey,
    MigrationSet,
    RenameEvent,
    backfill,
    migrate_from,
    split_event,
    transform_fields,
)


def _drop_total(kw: dict[str, Any]) -> dict[str, Any]:
    kw.pop("total", None)
    return kw


def _keep(kw: dict[str, Any]) -> dict[str, Any]:
    return kw


class Chain(Namespace):
    @migrate_from("Chain.Old", "Chain.Mid")
    class Placed(DomainEvent):
        note: str = ""


class Pair(Namespace):
    @migrate_from("Pair.Old", backfill={"kind": "a", "source": "b"})
    class Approved(DomainEvent):
        kind: str
        source: str


class Filled(Namespace):
    @backfill("status", default="open")
    class Ordered(DomainEvent):
        status: str


class Reshaped(Namespace):
    @transform_fields(_drop_total)
    class Ordered(DomainEvent):
        note: str = ""


class OriginReshaped(Namespace):
    @migrate_from("OriginReshaped.Old", transform=_keep)
    class Ordered(DomainEvent):
        note: str = ""


class Anonymous(Namespace):
    @transform_fields(lambda kw: kw)
    class Ordered(DomainEvent):
        note: str = ""


class Partial(Namespace):
    @transform_fields(functools.partial(_drop_total))
    class Ordered(DomainEvent):
        note: str = ""


def _pick_side(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    return (Split.Right, kw) if kw.get("note") == "right" else None


class Split(Namespace):
    class Right(DomainEvent):
        note: str = ""

    @split_event(_pick_side, targets=(Right,))
    class Left(DomainEvent):
        note: str = ""


class Hand(Namespace):
    class Ordered(DomainEvent):
        status: str


_HAND_MIGRATIONS = (
    Migration(
        (
            RenameEvent("legacy.orders", "Order", __name__, "Hand.Ordered"),
            AddField(__name__, "Hand.Ordered", "status", default="open"),
        ),
        name="hand",
    ),
)


def _labels(*namespaces: type, migrations: tuple[Migration, ...] = ()) -> set[str]:
    serde = NamespaceAwareSerde(migrations, namespaces=namespaces)
    return {key.label for key in serde.client_migrations().keys}


def describe_client_migrations():
    def when_no_client_migration_exists():
        def it_holds_no_framework_migration():
            serde = NamespaceAwareSerde()

            assert serde.client_migrations().keys == frozenset()

    def when_migrate_from_declares_a_chain():
        def it_gives_one_rename_key_per_declared_source():
            assert _labels(Chain) == {
                "rename test_migration_keys:Chain.Old",
                "rename test_migration_keys:Chain.Mid",
            }

        def it_maps_each_source_to_the_live_class():
            serde = NamespaceAwareSerde(namespaces=(Chain,))

            assert dict(serde.client_migrations().renames) == {
                ("test_migration_keys", "Chain.Old"): (
                    "test_migration_keys",
                    "Chain.Placed",
                ),
                ("test_migration_keys", "Chain.Mid"): (
                    "test_migration_keys",
                    "Chain.Placed",
                ),
            }

    def when_migrate_from_has_a_backfill_pair():
        def it_gives_one_rename_and_two_origin_fills():
            assert _labels(Pair) == {
                "rename test_migration_keys:Pair.Old",
                "fill test_migration_keys:Pair.Old[kind]",
                "fill test_migration_keys:Pair.Old[source]",
            }

    def when_backfill_decorates_a_class():
        def it_gives_a_class_fill():
            assert _labels(Filled) == {
                "fill test_migration_keys:Filled.Ordered[status]"
            }

    def when_transform_fields_decorates_a_class():
        def it_gives_a_class_transform_via_the_function():
            assert _labels(Reshaped) == {
                "transform test_migration_keys:Reshaped.Ordered"
                " via test_migration_keys:_drop_total"
            }

    def when_migrate_from_has_a_transform():
        def it_gives_an_origin_transform_via_the_function():
            assert _labels(OriginReshaped) == {
                "rename test_migration_keys:OriginReshaped.Old",
                "transform test_migration_keys:OriginReshaped.Old"
                " via test_migration_keys:_keep",
            }

    def when_the_transform_is_a_lambda():
        def it_names_the_lambda_by_its_qualname():
            assert _labels(Anonymous) == {
                "transform test_migration_keys:Anonymous.Ordered"
                " via test_migration_keys:Anonymous.<lambda>"
            }

    def when_the_transform_is_a_callable_object():
        def it_names_an_instance_of_its_class():
            assert _labels(Partial) == {
                "transform test_migration_keys:Partial.Ordered via functools:partial()"
            }

    def when_split_event_decorates_a_class():
        def it_gives_a_split_via_the_select_function():
            assert _labels(Split) == {
                "split test_migration_keys:Split.Left"
                " via test_migration_keys:_pick_side"
            }

    def when_migrations_is_a_hand_written_list():
        def it_gives_one_key_per_operation():
            assert _labels(Hand, migrations=_HAND_MIGRATIONS) == {
                "rename legacy.orders:Order",
                "fill test_migration_keys:Hand.Ordered[status]",
            }


def describe_MigrationKey():
    def describe_parse():
        @pytest.mark.parametrize(
            "label",
            [
                "rename legacy.orders:Order",
                "fill test_migration_keys:Pair.Old[kind]",
                "transform test_migration_keys:Reshaped.Ordered"
                " via test_migration_keys:_drop_total",
                "split test_migration_keys:Split.Left"
                " via test_migration_keys:outer.<locals>.<lambda>",
            ],
        )
        def it_inverts_the_label(label: str):
            assert MigrationKey.parse(label).label == label

        def it_reads_each_part_of_a_fill():
            key = MigrationKey.parse("fill app.events:Order[status]")

            assert key == MigrationKey("fill", "app.events", "Order", field="status")

        def it_reads_each_part_of_a_transform():
            key = MigrationKey.parse("transform app.events:Order via app.fix:_drop")

            assert key == MigrationKey(
                "transform", "app.events", "Order", via="app.fix:_drop"
            )

        @pytest.mark.parametrize(
            "label",
            [
                "",
                "rename",
                "move app.events:Order",
                "rename app.events.Order",
                "fill app.events:Order",
                "transform app.events:Order",
                "rename app.events:Order via app.fix:_drop",
                "split app.events:Order[status] via app.fix:_pick",
            ],
        )
        def it_refuses_a_malformed_label(label: str):
            with pytest.raises(ValueError, match="migration label"):
                MigrationKey.parse(label)

    def describe_construction():
        @pytest.mark.parametrize(
            ("kind", "field", "via"),
            [
                ("rename", "status", None),
                ("rename", None, "app.fix:_drop"),
                ("fill", None, None),
                ("fill", "status", "app.fix:_drop"),
                ("transform", None, None),
                ("transform", "status", "app.fix:_drop"),
                ("split", None, None),
            ],
        )
        def it_refuses_a_part_the_kind_does_not_have(
            kind: Any, field: str | None, via: str | None
        ):
            with pytest.raises(ValueError, match="MigrationKey"):
                MigrationKey(kind, "app.events", "Order", field=field, via=via)

        @pytest.mark.parametrize(
            ("module", "qualname", "field"),
            [
                ("app.events", "Order[int]", None),
                ("app[1]", "Order", None),
                ("app.events", "Order via x", None),
                ("app.events", "Order", "status]"),
                ("old:pkg", "Gone", None),
                ("app.events", "Gone via", None),
                ("", "Order", None),
                ("app.events", "", None),
                ("app events", "Order", None),
            ],
        )
        def it_refuses_a_part_the_label_cannot_hold(
            module: str, qualname: str, field: str | None
        ):
            kind = "fill" if field else "rename"
            with pytest.raises(ValueError, match="MigrationKey"):
                MigrationKey(kind, module, qualname, field=field)

    def describe_anonymous():
        def when_via_names_no_stable_callable():
            @pytest.mark.parametrize(
                "via",
                ["app.fix:<lambda>", "app.fix:Order.<lambda>", "functools:partial()"],
            )
            def it_is_anonymous(via: str):
                key = MigrationKey("transform", "app.events", "Order", via=via)
                assert key.anonymous

        def when_via_names_a_function():
            def it_is_not_anonymous():
                key = MigrationKey("split", "app.events", "Order", via="app.fix:_pick")
                assert not key.anonymous

        def when_the_key_has_no_via():
            def it_is_not_anonymous():
                assert not MigrationKey("rename", "app.events", "Order").anonymous

        def when_via_is_empty():
            def it_refuses_the_key():
                with pytest.raises(ValueError, match="MigrationKey"):
                    MigrationKey("transform", "app.events", "Order", via="")


def _keys(*labels: str) -> frozenset[MigrationKey]:
    return frozenset(MigrationKey.parse(label) for label in labels)


def _set(*labels: str, renames: dict[str, str] | None = None) -> MigrationSet:
    def identity(text: str) -> tuple[str, str]:
        module, _, qualname = text.partition(":")
        return module, qualname

    return MigrationSet(
        _keys(*labels),
        {identity(s): identity(t) for s, t in (renames or {}).items()},
    )


def describe_MigrationSet():
    def describe_fingerprint():
        def it_hashes_the_sorted_labels_under_a_versioned_prefix():
            fingerprint = _set("fill m:A[x]", "rename m:Old").fingerprint

            assert fingerprint == "lge-m1-f178eae2be39f503"

        def it_ignores_the_order_of_the_keys():
            first = _set("fill m:A[x]", "rename m:Old").fingerprint
            second = _set("rename m:Old", "fill m:A[x]").fingerprint

            assert first == second

        def it_gives_an_empty_set_a_fingerprint():
            assert _set().fingerprint == "lge-m1-e3b0c44298fc1c14"

    def describe_shipped_in():
        def when_a_previous_key_is_still_in_the_code():
            def it_reports_the_key():
                current = _set("fill m:A[x]", "fill m:B[y]")

                shipped = current.shipped_in(_keys("fill m:A[x]"))

                assert shipped == _keys("fill m:A[x]")

        def when_a_class_stage_key_follows_a_current_rename():
            def it_reports_the_moved_key():
                current = _set(
                    "rename m:X", "transform m:Y via m:_f", renames={"m:X": "m:Y"}
                )

                shipped = current.shipped_in(_keys("transform m:X via m:_f"))

                assert shipped == _keys("transform m:Y via m:_f")

        def when_an_origin_stage_key_was_deleted():
            def it_does_not_follow_the_rename_to_a_class_fill():
                current = _set(
                    "rename m:Old", "fill m:New[f]", renames={"m:Old": "m:New"}
                )

                shipped = current.shipped_in(_keys("rename m:Old", "fill m:Old[f]"))

                assert shipped == _keys("rename m:Old")

        def when_no_previous_key_is_in_the_code():
            def it_reports_nothing():
                assert _set("fill m:A[x]").shipped_in(_keys("fill m:B[y]")) == set()

    def describe_deleted_since():
        def when_a_previous_key_left_the_code():
            def it_reports_the_previous_key():
                current = _set("fill m:A[x]")

                deleted = current.deleted_since(_keys("fill m:A[x]", "rename m:Old"))

                assert deleted == _keys("rename m:Old")

        def when_a_previous_key_moved_through_a_current_rename():
            def it_reports_nothing():
                current = _set(
                    "rename m:X", "transform m:Y via m:_f", renames={"m:X": "m:Y"}
                )

                assert current.deleted_since(_keys("transform m:X via m:_f")) == set()

        def when_an_origin_stage_key_was_deleted():
            def it_reports_the_origin_key():
                current = _set(
                    "rename m:Old", "fill m:New[f]", renames={"m:Old": "m:New"}
                )

                deleted = current.deleted_since(_keys("rename m:Old", "fill m:Old[f]"))

                assert deleted == _keys("fill m:Old[f]")
