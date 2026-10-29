"""Namespaces, targets and two-release fixtures shared by the migration
tests: the gate, the baseline update and the CLI."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Interrupted,
    Namespace,
)
from langgraph_events.agui import FrontendToolCallRequested
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    Migration,
    MigrationKey,
    MigrationTarget,
    backfill,
    migrate_from,
    split_event,
    transform_fields,
    update_baseline,
)

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


class TrimmedWide(Namespace):
    class Place(Command):
        note: str = ""

        @transform_fields(_drop_total)
        class Placed(DomainEvent):
            note: str = ""
            legacy_b: str = ""

        def handle(self) -> TrimmedWide.Place.Placed:
            return TrimmedWide.Place.Placed(note=self.note)


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


def _deleted_transform(
    tmp_path: Path, shipped: type[Namespace] = Trimmed
) -> tuple[MigrationTarget, Path]:
    """Release N held a transform on ``Shop.Place.Placed`` that dropped
    ``total``. *shipped* is the namespace of release N. The code of release
    N+1 deleted the transform. Returns the N+1 target, whose committed
    baseline is still N's, and N's baseline."""
    release = _target(tmp_path, shipped, name="previous.json")
    update_baseline(release)
    raw = json.loads(release.baseline.read_text().replace(shipped.__name__, "Shop"))
    placed = raw["events"][1]
    placed["fields"] = sorted({*placed["fields"], "total"})
    release.baseline.write_text(json.dumps(raw))
    target = _target(tmp_path, Shop)
    target.baseline.write_text(release.baseline.read_text())
    return target, release.baseline


_OLD_TRIMMED = f"transform {_MODULE}:Trimmed.Place.Placed via {_MODULE}:_old_drop"


def _changed_key(tmp_path: Path) -> tuple[MigrationTarget, Path]:
    """Release N records ``Trimmed.Place.Placed`` with ``total`` and a
    transform keyed ``_old_drop``. The code keeps a transform on the same
    identity under the key ``_drop_total``. Returns the N+1 target, whose
    committed baseline is still N's, and N's baseline."""
    target = _target(tmp_path, Trimmed)
    update_baseline(target)
    raw = _read(target.baseline)
    raw["migrations"] = [{"label": _OLD_TRIMMED}]
    placed = raw["events"][1]
    placed["fields"] = sorted({*placed["fields"], "total"})
    target.baseline.write_text(json.dumps(raw))
    previous = tmp_path / "previous.json"
    previous.write_text(target.baseline.read_text())
    return target, previous


def _moved_transform(tmp_path: Path) -> MigrationTarget:
    """The committed baseline records the transform of ``_drop_total`` on
    ``Old.Placed``. The code renamed ``Old.Placed`` to
    ``Trimmed.Place.Placed``, and the transform followed the class."""
    target = _target(
        tmp_path,
        Trimmed,
        migrations=(
            Migration.rename(
                old_module=_MODULE, old_qualname="Old.Placed", to=Trimmed.Place.Placed
            ),
        ),
    )
    update_baseline(target)
    raw = _read(target.baseline)
    raw["migrations"] = [{"label": f"transform {_MODULE}:Old.Placed via {_DROP_TOTAL}"}]
    target.baseline.write_text(json.dumps(raw))
    return target


def _baselined(target: MigrationTarget) -> MigrationTarget:
    update_baseline(target)
    return target
