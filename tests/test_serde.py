"""Tests for ``langgraph_events.serde`` — namespace-aware checkpoint serde."""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from typing import Any, ClassVar

import ormsgpack
import pytest
from conftest import Started
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Halted,
    IntegrationEvent,
    Interrupted,
    Namespace,
    Resumed,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde, synthesize_legacy_payload

# ``_legacy_interrupted_payload`` and the inline Scatter-union test below
# still assemble Interrupt-wrapped / list payloads by hand, which the
# public ``synthesize_legacy_payload`` (single top-level event) doesn't
# cover. This is the one intentional private-symbol import in the suite.
from langgraph_events.serde._jsonplus import (
    EXT_INTERRUPT,
    EXT_NAMESPACE_AWARE_EVENT,
    UnrevivedIdentity,
    _option,
)
from langgraph_events.serde.migrations import (
    backfill,
    migrate_from,
    split_event,
    transform_fields,
)


def _baseline_file(tmp_path: Any, *identities: tuple[str, str]) -> Any:
    """Write a v1 baseline JSON listing *identities* and return its path."""
    import json

    target = tmp_path / "baseline.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {"module": module, "qualname": qualname}
                    for module, qualname in identities
                ],
            }
        )
    )
    return target


def _baseline_v3_file(
    tmp_path: Any,
    *,
    events: tuple[tuple[str, str, list[str] | None], ...] = (),
    retired: tuple[tuple[str, str, list[str] | None], ...] = (),
) -> Any:
    """Write a v3 baseline JSON and return its path.

    Each entry is ``(module, qualname, fields)``. ``fields`` is ``None`` for
    a retired entry whose record predates v3.
    """
    import json

    def entry(module: str, qualname: str, fields: list[str] | None) -> dict[str, Any]:
        record: dict[str, Any] = {"module": module, "qualname": qualname}
        if fields is not None:
            record["fields"] = fields
        return record

    target = tmp_path / "baseline.json"
    target.write_text(
        json.dumps(
            {
                "version": 3,
                "events": [entry(*e) for e in events],
                "retired": [entry(*r) for r in retired],
            }
        )
    )
    return target


def _build_decoreorg_graph() -> Any:
    """Factory referenced by the detect-CLI tests via ``module:attr``."""
    return EventGraph.from_namespaces(DecoReorg)


def _legacy_interrupted_payload(
    module: str, qualname: str, kwargs: dict[str, Any], interrupt_id: str
) -> tuple[str, bytes]:
    """Synthesize an ``Interrupt(value=<legacy event>, id=...)`` payload.

    LangGraph wraps interrupted values in :class:`Interrupt` before
    checkpointing, so HITL flows store renamed events under EXT_INTERRUPT
    with the event nested at one extra layer. Migrations must reach that
    nested layer through the ext-hook's recursion."""
    nested_event_inner = ormsgpack.packb((module, qualname, kwargs), option=_option)
    nested_event_ext = ormsgpack.Ext(EXT_NAMESPACE_AWARE_EVENT, nested_event_inner)
    interrupt_inner = ormsgpack.packb((nested_event_ext, interrupt_id), option=_option)
    outer = ormsgpack.packb(
        ormsgpack.Ext(EXT_INTERRUPT, interrupt_inner), option=_option
    )
    return ("msgpack", outer)


# Two namespaces that intentionally share leaf event names. With the default
# ``JsonPlusSerializer`` (which keys class identity by ``cls.__name__``),
# round-tripping ``Persona.Approve.Approved`` and ``Story.Approve.Approved``
# collides — both encode as ``"Approved"``. The namespace-aware serde keys by
# ``__qualname__`` instead, so the two stay distinguishable.
class Persona(Namespace):
    class Approve(Command):
        note: str = ""

        class Approved(DomainEvent):
            note: str = ""

        def handle(self) -> Persona.Approve.Approved:
            return Persona.Approve.Approved(note=self.note)


class Story(Namespace):
    class Approve(Command):
        note: str = ""

        class Approved(DomainEvent):
            note: str = ""

        def handle(self) -> Story.Approve.Approved:
            return Story.Approve.Approved(note=self.note)


# Fixture for the field-shape diagnostics. ``reason`` is required and is the
# only field, so one class covers both mismatch directions below.
class FieldShape(Namespace):
    class Persisted(DomainEvent):
        reason: str


# Fixture for #172. ``derived`` is ``init=False`` and is rebuilt in
# ``__post_init__``. The serde must not write it: the constructor rejects it.
class DerivedField(Namespace):
    class Placed(DomainEvent):
        total: int
        derived: int = dataclasses.field(init=False, default=0)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", self.total * 2)


# Fixture for the revive gate under #172. ``derived`` is ``init=False`` with
# no default. The placeholder helper must not fill it: it is not a kwarg.
# ``__post_init__`` does not read ``total``, so the ``None`` placeholder for
# ``total`` is accepted, as the revive gate documents.
class DerivedNoDefault(Namespace):
    class Placed(DomainEvent):
        total: int
        derived: int = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", 2)


# Fixtures for the decorator-driven migration tests. Module-level so their
# qualnames resolve through ``importlib.import_module`` + dotted ``getattr``
# at serde-construction validation time. Function-local classes acquire
# ``<locals>`` segments in their ``__qualname__`` and would be rejected as
# unresolvable migration chain targets.
class DecoReorg(Namespace):
    class Persist(Command):
        note: str = ""

        @migrate_from("DecoReorg.Persisted")
        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> DecoReorg.Persist.Persisted:
            return DecoReorg.Persist.Persisted(note=self.note)


# Fixture for the resolution-only gate. ``name`` is required and
# ``__post_init__`` rejects an empty/None value — mirrors agui's
# ``FrontendToolCallRequested``. ``assert_all_baselined_revive`` constructs the
# live class with a ``None`` placeholder and trips this validation, whereas
# ``assert_all_baselined_resolve`` only resolves the identity and passes.
class ValidatedReorg(Namespace):
    class Persist(Command):
        name: str = "x"

        @migrate_from("ValidatedReorg.Persisted")
        class Persisted(DomainEvent):
            name: str

            def __post_init__(self) -> None:
                if not self.name:
                    raise ValueError("name must be non-empty")

        def handle(self) -> ValidatedReorg.Persist.Persisted:
            return ValidatedReorg.Persist.Persisted(name=self.name)


# Fixture for the v3 revive gate. ``name`` has a default and
# ``__post_init__`` rejects an empty/None value. The baseline records
# ``name``, the live class still accepts it, so the gate must send no
# placeholder for it: a ``None`` there trips the validator that a real
# payload never trips.
class ValidatedDefault(Namespace):
    class Persist(Command):
        name: str = "x"

        class Persisted(DomainEvent):
            name: str = "x"

            def __post_init__(self) -> None:
                if not self.name:
                    raise ValueError("name must be non-empty")

        def handle(self) -> ValidatedDefault.Persist.Persisted:
            return ValidatedDefault.Persist.Persisted(name=self.name)


# Fixture for an ``init=False`` field. The serde writes init fields only,
# so the baseline must record the same set. A recorded ``length`` would
# reach the constructor as a placeholder, and the constructor rejects it.
class Derived(Namespace):
    class Persist(Command):
        note: str = ""

        class Persisted(DomainEvent):
            note: str = ""
            length: int = dataclasses.field(init=False)

            def __post_init__(self) -> None:
                object.__setattr__(self, "length", len(self.note))

        def handle(self) -> Derived.Persist.Persisted:
            return Derived.Persist.Persisted(note=self.note)


# Tombstones for identities a baseline lists under ``retired``. The
# retired ``Retiring.ApprovalRequired`` recorded ``order_id``, so its
# tombstone declares the same field. ``Retiring.ReviewRequired`` predates
# v3 (no field record), and its tombstone has a required field the gate
# must placeholder.
class Retiring(Namespace):
    @migrate_from("Retiring.ApprovalRequired")
    class RetiredApprovalGate(Interrupted):
        order_id: str = ""

    @migrate_from("Retiring.ReviewRequired")
    class RetiredReviewGate(Interrupted):
        order_id: str


class DecoMulti(Namespace):
    class Persist(Command):
        note: str = ""

        @migrate_from("DecoMulti.Persisted", "DecoMulti.OldNest.Persisted")
        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> DecoMulti.Persist.Persisted:
            return DecoMulti.Persist.Persisted(note=self.note)


# Fixture for the stacked-decorator order test. Python applies decorators
# bottom-up, so ``@migrate_from("DecoStack.Oldest")`` runs first; the
# resulting history must place it ahead of the outer decorator's args.
class DecoStack(Namespace):
    class Persist(Command):
        note: str = ""

        @migrate_from("DecoStack.Newer")
        @migrate_from("DecoStack.Oldest")
        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> DecoStack.Persist.Persisted:
            return DecoStack.Persist.Persisted(note=self.note)


# Fixture for the subclass-leak tests. ``Parent`` carries ``@migrate_from``;
# ``Child`` deliberately does NOT — the bug is that ``Child`` inherits the
# attribute through Python's normal MRO lookup, which leaks into both the
# auto-collection path (duplicate Migration entry → "Duplicate rename
# source" at serde construction) and ``legacy_write`` (writes Child
# instances under the parent's historic qualname). Subclassing a
# DomainEvent across a Namespace level is allowed by the framework.
class SubLeak(Namespace):
    @migrate_from("SubLeak.OldParent")
    class Parent(DomainEvent):
        note: str = ""

    class Child(Parent):
        extra: str = ""


# Fixtures for the scoped-collection tests. ``ScopeFixtureA`` carries a
# valid ``@migrate_from`` decoration; ``ScopeFixtureB`` does not. A serde
# scoped to ``[ScopeFixtureB]`` must not collect ``ScopeFixtureA``'s
# migration, and must not relabel ``ScopeFixtureA`` instances under their
# historic qualname even with ``legacy_write=True``.
class ScopeFixtureA(Namespace):
    class Persist(Command):
        note: str = ""

        @migrate_from("ScopeFixtureA.Persisted")
        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> ScopeFixtureA.Persist.Persisted:
            return ScopeFixtureA.Persist.Persisted(note=self.note)


class ScopeFixtureB(Namespace):
    class Persist(Command):
        note: str = ""

        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> ScopeFixtureB.Persist.Persisted:
            return ScopeFixtureB.Persist.Persisted(note=self.note)


# Fixture for AddField migration tests. ``command_id`` is required (no
# default) — old payloads written before the field existed will fail to
# construct unless an ``AddField`` operation injects a value into kwargs.
# ``tags`` defaults to an empty tuple so we can also exercise the
# ``default_factory`` path without dataclass field-ordering pain.
class NestedPersist(Namespace):
    class Persist(Command):
        note: str = ""

        class Persisted(DomainEvent):
            command_id: str
            note: str = ""
            tags: tuple[str, ...] = ()

        def handle(self) -> NestedPersist.Persist.Persisted:
            return NestedPersist.Persist.Persisted(command_id="cid-1", note=self.note)


# Fixtures for @backfill (Finding 2b). ``command_id`` is required in new
# code but pre-existing payloads predate it — @backfill injects the legacy
# value only on the deserialization path, with no migrations= list and no
# manual serde. ``DecoBackfillRenamed`` also carries @migrate_from to prove
# the two class-scoped decorators compose.
class DecoBackfill(Namespace):
    class Persist(Command):
        note: str = ""

        @backfill("command_id", default="legacy")
        class Persisted(DomainEvent):
            command_id: str
            note: str = ""

        def handle(self) -> DecoBackfill.Persist.Persisted:
            return DecoBackfill.Persist.Persisted(command_id="cid", note=self.note)


class DecoBackfillRenamed(Namespace):
    class Persist(Command):
        note: str = ""

        @migrate_from("DecoBackfillRenamed.Persisted")
        @backfill("command_id", default="legacy")
        class Persisted(DomainEvent):
            command_id: str
            note: str = ""

        def handle(self) -> DecoBackfillRenamed.Persist.Persisted:
            return DecoBackfillRenamed.Persist.Persisted(
                command_id="cid", note=self.note
            )


# @backfill must not leak through MRO: Child does NOT redeclare it and
# must not inherit Parent's injected field.
class BackfillSubLeak(Namespace):
    @backfill("command_id", default="legacy")
    class Parent(DomainEvent):
        command_id: str
        note: str = ""

    class Child(Parent):
        extra: str = ""


# Stacked @backfill must accumulate — both fields injected for old payloads.
class DecoBackfillStack(Namespace):
    class Persist(Command):
        note: str = ""

        @backfill("flag", default=False)
        @backfill("command_id", default="legacy")
        class Persisted(DomainEvent):
            command_id: str
            flag: bool
            note: str = ""

        def handle(self) -> DecoBackfillStack.Persist.Persisted:
            return DecoBackfillStack.Persist.Persisted(
                command_id="cid", flag=True, note=self.note
            )


# A mutable @backfill default must be rejected by the SAME guard that
# rejects AddField(default=[]) — not a forked rule.
class DecoBackfillBadDefault(Namespace):
    @backfill("tags", default=[])  # exercising the mutable-default guard
    class Persisted(DomainEvent):
        note: str = ""


# Fixture for origin-scoped backfill (#101): three per-entity classes
# collapsed into one shared class with a required ``entity_type``
# discriminator the old payloads never carried. ``backfill=`` on each
# ``@migrate_from`` pins the value for payloads that originated under THAT
# qualname — a class-global ``@backfill`` could only inject one value for
# all three origins.
class DecoFanIn(Namespace):
    @migrate_from("PersonaOld.Approve.Approved", backfill={"entity_type": "persona"})
    @migrate_from("StoryOld.Approve.Approved", backfill={"entity_type": "story"})
    @migrate_from("ScenarioOld.Approve.Approved", backfill={"entity_type": "scenario"})
    class Approved(DomainEvent):
        entity_type: str
        entity_id: str = ""


# An origin without a scoped fill falls back to the class-global @backfill
# — scoped fills override the fallback only for THEIR origin.
class DecoFanInFallback(Namespace):
    @backfill("entity_type", default="unknown")
    @migrate_from("PersonaOldFB.Approved", backfill={"entity_type": "persona"})
    @migrate_from("LegacyOldFB.Approved")
    class Approved(DomainEvent):
        entity_type: str
        entity_id: str = ""


# Origin-scoped fills must not leak through MRO: Child does NOT redeclare
# the decorator and must not re-emit Parent's origin AddField (which the
# duplicate-fill guard would reject at serde construction).
class OriginBackfillSubLeak(Namespace):
    @migrate_from("OriginBackfillSubLeak.OldParent", backfill={"command_id": "legacy"})
    class Parent(DomainEvent):
        command_id: str
        note: str = ""

    class Child(Parent):
        extra: str = ""


# A back-fill naming a field the live class doesn't have must fail at
# serde construction, not as a TypeError at first production read.
class DecoFanInTypoField(Namespace):
    @migrate_from("TypoOld.Approved", backfill={"entity_typ": "persona"})
    class Approved(DomainEvent):
        entity_type: str
        entity_id: str = ""


# Same guard for the class-global decorator — the gap predates origin
# scoping but the fix covers both forms through one AddField check.
class DecoBackfillTypoField(Namespace):
    @backfill("command_idd", default="legacy")
    class Persisted(DomainEvent):
        command_id: str
        note: str = ""


# Fixture for #172. A fill on an ``init=False`` field is not a kwarg the
# constructor accepts, so the serde must refuse it at construction.
class DecoBackfillInitFalse(Namespace):
    @backfill("derived", default=0)
    class Placed(DomainEvent):
        total: int
        derived: int = dataclasses.field(init=False, default=0)


# Fixture proving the revive gate exercises origin fills instead of
# masking them with ``None`` placeholders: ``__post_init__`` rejects
# anything but a real discriminator, so the gate only passes if the
# origin-scoped value actually flows through the read path.
class ValidatedFanIn(Namespace):
    @migrate_from("ValidatedFanInOld.Approved", backfill={"entity_type": "persona"})
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story", "scenario"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


# Class-global analog of ValidatedFanIn: __post_init__ validation on a
# class-globally back-filled field must pass the revive gate (the fill's
# real value is injected, not a None placeholder).
class ValidatedGlobalBackfill(Namespace):
    @backfill("entity_type", default="persona")
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story", "scenario"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


# Negative twin: a fill whose value violates construction-time validation
# must FAIL the gate — the placeholder skip exists precisely so a broken
# fill cannot hide behind a None placeholder.
class ValidatedBadFill(Namespace):
    @backfill("entity_type", default="bogus")
    class Approved(DomainEvent):
        entity_type: str

        def __post_init__(self) -> None:
            if self.entity_type not in ("persona", "story", "scenario"):
                raise ValueError(f"bad entity_type: {self.entity_type!r}")


# Fixture for the raw-form tests: a temporal chain whose NEWER historic
# identity carries a hand-authored origin-keyed AddField. Payloads from
# the OLDER era must not pick it up — origin scoping is an exact match,
# not chain-aware.
class ChainScoped(Namespace):
    @migrate_from("ChainScoped.Old", "ChainScoped.New")
    class Persisted(DomainEvent):
        source: str = ""


# Fixture for the per-origin default_factory escape hatch: ``tags`` is
# required and list-typed, which ``backfill=`` (immutable values only)
# cannot express — the raw AddField keyed on the historic identity can.
class FactoryScoped(Namespace):
    @migrate_from("FactoryScoped.Old")
    class Persisted(DomainEvent):
        tags: list[str]
        note: str = ""


# Fixture that carries one of every member shape a ``Namespace`` body can
# hold, so ``revivable_identities`` can be pinned as an exact set rather
# than by membership. Set membership is what makes an OVER-BROAD identity
# set invisible, and an over-broad set is the direction that silently
# weakens ``assert_all_baselined_cover`` (#154). Deliberately carries no
# ``@migrate_from``: the walk and the rename table are pinned separately.
class WalkShapes(Namespace):
    # Not a class at all — the walk must step over it.
    LABEL = "not-an-event"

    # A class that is not an ``Event`` — likewise stepped over.
    class Helper:
        pass

    # DomainEvent nested directly in the Namespace.
    class Recorded(DomainEvent):
        note: str = ""

    # Non-DomainEvent event nested in the Namespace for locality. The
    # metaclass never fires for these, so only the walk reaches them.
    class Blocked(Halted):
        reason: str = ""

    # Single-outcome Command: ``Outcomes`` is the nested class itself, the
    # alias ``_iter_nested_events`` skips so it is not re-visited.
    class Record(Command):
        note: str = ""

        # Reached only because the walk recurses into Commands.
        class Stored(DomainEvent):
            note: str = ""

        # Non-DomainEvent nested inside a Command — same recursion.
        class Stalled(Halted):
            reason: str = ""

    # Multi-outcome Command: ``Outcomes`` is a ``UnionType``, not a class.
    class Amend(Command):
        note: str = ""

        class Amended(DomainEvent):
            note: str = ""

        class Rejected(DomainEvent):
            reason: str = ""


# Namespace whose nested ``Reviewed`` is an ``Interrupted`` subclass. This is
# the shape that hits #60 — handlers return ``Review.Ask.Reviewed(...)`` and
# LangGraph wraps it in ``langgraph.types.Interrupt(value=..., id=...)`` for
# the checkpoint write.
class Review(Namespace):
    class Ask(Command):
        draft: str = ""

        class Reviewed(Interrupted):
            draft: str = ""

        def handle(self) -> Review.Ask.Reviewed:
            return Review.Ask.Reviewed(draft=self.draft)


# Module-level so the ``resume`` handler's return annotation is resolvable by
# ``typing.get_type_hints`` against the module globals. Defining it inside the
# test function would trigger an "unresolved type hints" warning at handler
# registration.
class ReviewApproved(IntegrationEvent):
    pass


# Plain (non-Event) dataclass — exercises the upstream
# ``EXT_CONSTRUCTOR_KW_ARGS`` revival path that #68 broke. Module-level so
# the qualname registered at encode time resolves on the import-and-getattr
# walk performed by upstream's ext-hook.
def _rename_migration(
    name: str,
    *,
    old_module: str,
    old_qualname: str,
    new_module: str,
    new_qualname: str,
) -> Any:
    """A single-operation ``Migration`` that renames one event identity."""
    from langgraph_events.serde.migrations import Migration, RenameEvent

    return Migration(
        name=name,
        operations=(
            RenameEvent(
                old_module=old_module,
                old_qualname=old_qualname,
                new_module=new_module,
                new_qualname=new_qualname,
            ),
        ),
    )


def _persona_rename(name: str, **overrides: Any) -> Any:
    """The recurring ``legacy.persona:Legacy.Approved`` -> ``Persona.Approve.Approved``
    rename; ``overrides`` swap individual endpoints."""
    endpoints: dict[str, Any] = {
        "old_module": "legacy.persona",
        "old_qualname": "Legacy.Approved",
        "new_module": Persona.__module__,
        "new_qualname": "Persona.Approve.Approved",
    }
    endpoints.update(overrides)
    return _rename_migration(name, **endpoints)


def _add_field_migration(
    name: str, *, module: str, qualname: str, field: str, **default: Any
) -> Any:
    """A single-operation ``Migration`` that injects one missing field."""
    from langgraph_events.serde.migrations import AddField, Migration

    return Migration(
        name=name,
        operations=(
            AddField(module=module, qualname=qualname, field=field, **default),
        ),
    )


def _persisted_field_migration(name: str, field: str, **default: Any) -> Any:
    """``_add_field_migration`` keyed on ``NestedPersist.Persist.Persisted``."""
    return _add_field_migration(
        name,
        module=NestedPersist.__module__,
        qualname="NestedPersist.Persist.Persisted",
        field=field,
        **default,
    )


def _persisted_payload(**kwargs: Any) -> Any:
    """A legacy wire payload for ``NestedPersist.Persist.Persisted``."""
    return synthesize_legacy_payload(
        NestedPersist.__module__, "NestedPersist.Persist.Persisted", kwargs
    )


def _local_trading() -> Any:
    """A ``Namespace`` built inside a function — its qualname carries
    ``<locals>``, so no import walk can ever reach the nested classes."""

    class Trading(Namespace):
        class Place(Command):
            sym: str

            class Placed(DomainEvent):
                sym: str

    return Trading


def _local_migrated_trading() -> Any:
    """``_local_trading`` after a rename: the ``@migrate_from`` chain
    terminates on a class no import walk can reach."""

    class Trading(Namespace):
        class Place(Command):
            sym: str

            @migrate_from("Trading.Place.Filled")
            class Placed(DomainEvent):
                sym: str

    return Trading


def _local_backfilled_trading() -> Any:
    """``_local_trading`` after gaining a field: the ``@backfill`` target is
    a class no import walk can reach."""

    class Trading(Namespace):
        class Place(Command):
            sym: str

            @backfill("venue", default="NYSE")
            class Placed(DomainEvent):
                sym: str
                venue: str

    return Trading


@dataclasses.dataclass
class PlainPayload:
    name: str
    count: int


# Pydantic v2 model — separate ext-code branch (``EXT_PYDANTIC_V2``) than
# ``PlainPayload``. Pydantic-heavy state was the reported repro for #68.
class PydanticPayload(BaseModel):
    name: str
    count: int


class _ModelHolder:
    """Nests a pydantic model inside a plain class (#167). Upstream stores
    a pydantic payload by ``__name__`` and revives it with ``getattr`` on
    the module, so ``Cfg`` never resolves and revives as a raw dict."""

    class Cfg(BaseModel):
        tag: str = "x"


class _NestedPayloadNs(Namespace):
    class Placed(DomainEvent):
        cfg: _ModelHolder.Cfg


class _NestedPayloadInGenericNs(Namespace):
    class Placed(DomainEvent):
        cfgs: list[_ModelHolder.Cfg] | None = None


class _ModulePayloadNs(Namespace):
    class Placed(DomainEvent):
        cfg: PydanticPayload


class _ClassVarPayloadNs(Namespace):
    class Placed(DomainEvent):
        """A ClassVar is not a dataclass field and is never checkpointed."""

        default_cfg: ClassVar[_ModelHolder.Cfg] = _ModelHolder.Cfg()


class _ModelHolderV1:
    class Cfg(BaseModelV1):
        """pydantic v1 takes the EXT_PYDANTIC_V1 branch upstream, with the
        same getattr-by-__name__ miss."""

        tag: str = "x"


class _NestedV1PayloadNs(Namespace):
    class Placed(DomainEvent):
        cfg: _ModelHolderV1.Cfg


class _SiblingRefPayloadNs(Namespace):
    """A field typed as a sibling by forward reference. get_type_hints
    without the enclosing class's names raises NameError on it, and an
    all-or-nothing fallback would then skip the nested model beside it."""

    class Sibling(DomainEvent):
        pass

    class Placed(DomainEvent):
        other: Sibling | None = None  # noqa: F821 - resolved by the check's localns
        cfg: _ModelHolder.Cfg | None = None


def describe_NamespaceAwareSerde():
    def describe_dumps_typed_loads_typed_roundtrip():
        def when_two_namespaces_share_a_leaf_event_name():
            def it_preserves_class_identity_across_the_roundtrip():
                serde = NamespaceAwareSerde()
                p = Persona.Approve.Approved(note="persona")
                s = Story.Approve.Approved(note="story")

                p_back = serde.loads_typed(serde.dumps_typed(p))
                s_back = serde.loads_typed(serde.dumps_typed(s))

                assert isinstance(p_back, Persona.Approve.Approved)
                assert isinstance(s_back, Story.Approve.Approved)
                # Sibling identity is not collapsed.
                assert not isinstance(p_back, Story.Approve.Approved)
                assert not isinstance(s_back, Persona.Approve.Approved)
                assert p_back.note == "persona"
                assert s_back.note == "story"

        def when_an_event_has_an_init_false_field():
            def it_round_trips_and_recomputes_the_derived_value():
                serde = NamespaceAwareSerde(namespaces=(DerivedField,))
                ev = DerivedField.Placed(total=1)
                assert ev.derived == 2

                back = serde.loads_typed(serde.dumps_typed(ev))

                assert isinstance(back, DerivedField.Placed)
                assert back.total == 1
                assert back.derived == 2

    def describe_checkpoint_roundtrip():
        def when_two_namespaces_share_a_leaf_event_name():
            def it_restores_each_event_under_its_correct_class():
                checkpointer = MemorySaver(serde=NamespaceAwareSerde())
                graph = EventGraph(
                    [Persona.Approve, Story.Approve],
                    checkpointer=checkpointer,
                )

                p_config = {"configurable": {"thread_id": "p"}}
                s_config = {"configurable": {"thread_id": "s"}}
                graph.invoke(Persona.Approve(note="persona"), config=p_config)
                graph.invoke(Story.Approve(note="story"), config=s_config)

                p_state = graph.get_state(p_config)
                s_state = graph.get_state(s_config)

                p_event = p_state.events.latest(Persona.Approve.Approved)
                s_event = s_state.events.latest(Story.Approve.Approved)

                # If the serde collapsed identity, p_event would also satisfy
                # isinstance(s_event_class) — and vice versa.
                assert isinstance(p_event, Persona.Approve.Approved)
                assert isinstance(s_event, Story.Approve.Approved)
                assert not isinstance(p_event, Story.Approve.Approved)
                assert not isinstance(s_event, Persona.Approve.Approved)
                assert p_event.note == "persona"
                assert s_event.note == "story"

    def describe_revival_of_a_missing_class():
        def when_the_qualname_path_no_longer_resolves():
            def it_raises_a_clear_error_naming_the_missing_class():
                serde = NamespaceAwareSerde()

                # Encode a known class, then mutate the bytes so the qualname
                # references a path that doesn't exist on the module. Stand-in
                # for the real-world case: a checkpoint produced by an older
                # build whose Event class has since been renamed/removed.
                ev = Persona.Approve.Approved(note="ghost")
                kind, payload = serde.dumps_typed(ev)
                # Replace the qualname segment with a path that won't resolve.
                # Both b"Approved" and b"Approve" appear; replacing the rare
                # leaf is sufficient to break the attribute walk.
                tampered = payload.replace(b"Approved", b"GhostClass")
                assert tampered != payload

                with pytest.raises(ValueError, match=r"GhostClass|Persona\.Approve"):
                    serde.loads_typed((kind, tampered))

    def describe_revival_of_a_changed_class():
        # Sibling of the missing-class case above. The identity still
        # resolves, but the stored kwargs no longer match the live
        # ``__init__``, so the frozen dataclass raises ``TypeError``.

        def when_a_field_was_removed_from_the_class():
            def it_names_the_identity_and_the_rejected_field():
                serde = NamespaceAwareSerde()

                # An extra stored key reaches no absorber — ``__init__``
                # rejects every keyword the class does not declare.
                payload = synthesize_legacy_payload(
                    FieldShape.__module__,
                    "FieldShape.Persisted",
                    {"reason": "kept", "note": "dropped"},
                )

                with pytest.raises(ValueError) as excinfo:
                    serde.loads_typed(payload)

                message = str(excinfo.value)
                assert "FieldShape.Persisted" in message
                assert "TypeError" in message
                assert "note" in message

            def it_says_the_class_is_missing_the_field_not_to_migrate_from_it():
                # The identity resolved to a live class — possibly already
                # a tombstone carrying @migrate_from — so "map onto a
                # tombstone with @migrate_from" is nonsense advice here;
                # that remedy belongs to the ImportError/AttributeError
                # branch (identity doesn't resolve at all), not this one
                # (identity resolves, __init__ rejects a stored field).
                serde = NamespaceAwareSerde()

                payload = synthesize_legacy_payload(
                    FieldShape.__module__,
                    "FieldShape.Persisted",
                    {"reason": "kept", "note": "dropped"},
                )

                with pytest.raises(ValueError) as excinfo:
                    serde.loads_typed(payload)

                message = str(excinfo.value)
                assert "does not declare" in message
                assert "'note'" in message
                assert "@migrate_from" not in message

        def when_a_required_field_was_added_to_the_class():
            def without_a_migration():
                def it_names_the_identity_and_the_missing_field():
                    serde = NamespaceAwareSerde()

                    # A missing key CAN reach an absorber, but only a class
                    # default or an ``AddField`` supplies one. A required
                    # field has neither.
                    payload = synthesize_legacy_payload(
                        FieldShape.__module__, "FieldShape.Persisted", {}
                    )

                    with pytest.raises(ValueError) as excinfo:
                        serde.loads_typed(payload)

                    message = str(excinfo.value)
                    assert "FieldShape.Persisted" in message
                    assert "TypeError" in message
                    assert "reason" in message
                    assert "fields no longer match the stored payload" in message

        def when_the_stored_fields_still_match():
            def it_revives_the_event():
                serde = NamespaceAwareSerde()

                payload = synthesize_legacy_payload(
                    FieldShape.__module__,
                    "FieldShape.Persisted",
                    {"reason": "kept"},
                )

                revived = serde.loads_typed(payload)

                assert isinstance(revived, FieldShape.Persisted)
                assert revived.reason == "kept"

    def describe_encode_fallback():
        # ``NamespaceAwareSerde._default`` is a strict superset of upstream's
        # ``_msgpack_default`` — anything upstream encodes, we encode the
        # same way. A ``MsgpackEncodeError`` is therefore genuinely
        # unencodable; the warn-then-call-super path used to convert this
        # into a confusing two-stage failure (warning, then a re-raise from
        # the parent) in the default config, and into a silent unsafe-
        # binary emission when the parent's binary-fallback kwarg opted in
        # (bypassing the migration table). The contract is now: raise
        # loudly at the source, no fallback.

        def when_msgpack_encoding_fails():
            def without_parent_fallback_kwarg():
                def it_raises_msgpack_encode_error():
                    class Unencodable:
                        pass

                    serde = NamespaceAwareSerde()
                    with pytest.raises(ormsgpack.MsgpackEncodeError):
                        serde.dumps_typed(Unencodable())

                def it_does_not_emit_a_fallback_warning():
                    # The previous behavior warned before re-raising. With
                    # the fallback gone, the exception IS the signal — no
                    # warning fires.
                    class Unencodable:
                        pass

                    serde = NamespaceAwareSerde()
                    with (
                        warnings.catch_warnings(record=True) as caught,
                        contextlib.suppress(ormsgpack.MsgpackEncodeError),
                    ):
                        warnings.simplefilter("always")
                        serde.dumps_typed(Unencodable())

                    fallback_warnings = [
                        w
                        for w in caught
                        if "fall" in str(w.message).lower()
                        or "namespace" in str(w.message).lower()
                    ]
                    assert not fallback_warnings, (
                        f"expected no fallback warning; got: "
                        f"{[str(w.message) for w in fallback_warnings]}"
                    )

            def with_parent_fallback_kwarg_enabled():
                # Guard against the subtle alternative: if a user opts into
                # the parent JsonPlusSerializer's binary-fallback kwarg,
                # the parent's ``dumps_typed`` would silently emit unsafe
                # binary bytes. We bypass the parent entirely so namespace
                # state never gets opted into bytes the migration table
                # can't see. Kwarg name is passed via dict expansion to
                # keep the literal out of source readers' eyes.

                def it_still_raises_instead_of_emitting_unsafe_bytes():
                    class Unencodable:
                        pass

                    opt_in_unsafe_fallback = {"pickle_fallback": True}
                    serde = NamespaceAwareSerde(**opt_in_unsafe_fallback)
                    with pytest.raises(ormsgpack.MsgpackEncodeError):
                        serde.dumps_typed(Unencodable())

    def describe_unrevived_identity_on_write():
        # #170: ``UnrevivedIdentity`` is a read-side placeholder for the
        # retirement tools. Upstream's default hook would encode it as a
        # named tuple by class name, so it would round-trip as a stored
        # value a strict read cannot tell from a real event. The serde
        # must refuse to write it, whatever caller passes it.

        def when_a_placeholder_is_in_the_payload():
            def it_raises_naming_the_identity_and_the_remedy():
                serde = NamespaceAwareSerde()
                placeholder = UnrevivedIdentity(module="app.events", qualname="Gone")

                with pytest.raises(
                    ValueError,
                    match=(
                        r"app\.events\.Gone.*placeholder.*never.*stored.*"
                        r"tombstone.*Recovering a delete-first deployment"
                    ),
                ):
                    serde.dumps_typed([placeholder])

    def describe_event_nested_in_langgraph_Interrupt():
        # Regression for #60: every namespaced ``Interrupted`` subclass that
        # LangGraph wraps in ``langgraph.types.Interrupt(value=..., id=...)``
        # used to lose its identity through a checkpoint roundtrip — the
        # nested Event was encoded by upstream's ``_msgpack_default`` (leaf
        # ``__name__``) instead of our namespace-aware ext code, so the
        # decode-time attribute walk failed and was silently swallowed,
        # leaving ``Interrupt(value=None, id=...)``.
        def when_interrupt_wraps_a_namespaced_event():
            def it_preserves_the_nested_event_class_identity():
                serde = NamespaceAwareSerde()
                iv = Interrupt(
                    value=Persona.Approve.Approved(note="persona"),
                    id="abc123",
                )

                back = serde.loads_typed(serde.dumps_typed(iv))

                assert isinstance(back, Interrupt)
                assert back.id == "abc123"
                assert back.value is not None  # the bug: this used to be None
                assert isinstance(back.value, Persona.Approve.Approved)
                # Sibling identity is not collapsed across the roundtrip.
                assert not isinstance(back.value, Story.Approve.Approved)
                assert back.value.note == "persona"

        def when_a_checkpoint_carries_multiple_interrupts():
            # LangGraph's runner emits a *tuple* of ``Interrupt``s on a
            # checkpoint write (one per parallel branch / interrupt site).
            # Single-Interrupt round-trip is covered above; this guards the
            # actual emission shape so a regression that breaks tuple/list
            # iteration through ``_default`` shows up here.
            def it_preserves_each_nested_event_class_identity():
                serde = NamespaceAwareSerde()
                ivs = [
                    Interrupt(value=Persona.Approve.Approved(note="x"), id="a"),
                    Interrupt(value=Story.Approve.Approved(note="y"), id="b"),
                ]

                back = serde.loads_typed(serde.dumps_typed(ivs))

                assert len(back) == 2
                assert isinstance(back[0], Interrupt)
                assert isinstance(back[1], Interrupt)
                assert isinstance(back[0].value, Persona.Approve.Approved)
                assert isinstance(back[1].value, Story.Approve.Approved)
                # Sibling identity is not collapsed across either roundtrip.
                assert not isinstance(back[0].value, Story.Approve.Approved)
                assert not isinstance(back[1].value, Persona.Approve.Approved)
                assert back[0].id == "a"
                assert back[1].id == "b"
                assert back[0].value.note == "x"
                assert back[1].value.note == "y"

        def when_round_tripped_through_a_real_interrupt_flow():
            def with_MemorySaver_and_NamespaceAwareSerde():
                # ``filterwarnings("error", ...)`` locks the warning fix in:
                # if a future change demotes ``ReviewApproved`` back to a
                # local class, this test fails instead of silently emitting
                # the "Failed to resolve type hints" UserWarning.
                @pytest.mark.filterwarnings(
                    r"error:Failed to resolve.*type hints.*:UserWarning"
                )
                def it_round_trips_a_namespaced_Interrupted_subclass():
                    @on(Started)
                    def ask(event: Started) -> Review.Ask.Reviewed:
                        return Review.Ask.Reviewed(draft=event.data)

                    @on(Resumed, interrupted=Review.Ask.Reviewed)
                    def resume(
                        event: Resumed, interrupted: Review.Ask.Reviewed
                    ) -> ReviewApproved:
                        # The field-matcher only dispatches here if the
                        # round-tripped ``interrupted`` is a
                        # ``Review.Ask.Reviewed``. Before the fix it was
                        # ``None`` (Interrupt.value lost), the handler never
                        # fired, and ``ReviewApproved`` never appeared in
                        # the log.
                        assert isinstance(interrupted, Review.Ask.Reviewed)
                        assert interrupted.draft == "hello"
                        return ReviewApproved()

                    graph = EventGraph(
                        [ask, resume],
                        checkpointer=MemorySaver(serde=NamespaceAwareSerde()),
                    )
                    config = {"configurable": {"thread_id": "review"}}
                    graph.invoke(Started(data="hello"), config=config)

                    state = graph.get_state(config)
                    assert state.is_interrupted

                    log = graph.resume(ReviewApproved(), config=config)
                    assert log.latest(ReviewApproved) == ReviewApproved()

        def describe_Interrupt_schema_guard():
            # ``_default``/``_ext_hook`` track ``Interrupt`` by its two known
            # fields (``value``, ``id``). If LangGraph ever adds another
            # field, our hardcoded reconstruction would silently drop it on
            # round-trip — this guard surfaces that drift loudly so the
            # serde gets updated alongside the LangGraph bump.
            def it_matches_the_schema_we_encode():
                fields = {f.name for f in dataclasses.fields(Interrupt)}
                assert fields == {"value", "id"}, (
                    f"langgraph.types.Interrupt fields drifted from "
                    f"{{'value', 'id'}} to {fields}. NamespaceAwareSerde "
                    f"hardcodes (value, id) in _jsonplus.py — extend "
                    f"_default and the EXT_INTERRUPT branch of _ext_hook "
                    f"to cover the new field(s)."
                )

    def describe_non_event_payload_round_trip():
        # Regression for #68: ``langgraph-checkpoint>=4.0.3`` rebinds the
        # module-level ``_msgpack_ext_hook`` to a strict hook whose default
        # ``allowed_modules=None`` blocks everything outside
        # ``SAFE_MSGPACK_TYPES`` and silently demotes the value to a plain
        # ``dict``. Before the fix, ``NamespaceAwareSerde`` reached for that
        # module-level alias as the fallback for codes it didn't own — so
        # any non-event payload (Pydantic models, plain dataclasses, app
        # types) round-tripped as ``dict`` regardless of the constructor's
        # permissive default. The fix routes the fallback through the
        # parent's per-instance ``_unpack_ext_hook`` so the constructor's
        # allowlist (and ``LANGGRAPH_STRICT_MSGPACK``) take effect as
        # documented.
        def when_a_plain_dataclass_round_trips():
            def it_revives_as_the_original_class_not_a_dict():
                serde = NamespaceAwareSerde()
                obj = PlainPayload(name="x", count=42)

                back = serde.loads_typed(serde.dumps_typed(obj))

                assert isinstance(back, PlainPayload), (
                    f"expected PlainPayload, got {type(back).__name__}: {back!r}"
                )
                assert back.name == "x"
                assert back.count == 42

        def when_a_pydantic_v2_model_round_trips():
            # Different ext-code branch from ``PlainPayload``
            # (``EXT_PYDANTIC_V2``). Pydantic-heavy state is the realistic
            # production payload reported in #68.
            def it_revives_as_the_original_model_not_a_dict():
                serde = NamespaceAwareSerde()
                obj = PydanticPayload(name="x", count=42)

                back = serde.loads_typed(serde.dumps_typed(obj))

                assert isinstance(back, PydanticPayload), (
                    f"expected PydanticPayload, got {type(back).__name__}: {back!r}"
                )
                assert back.name == "x"
                assert back.count == 42

        def when_an_event_field_is_a_pydantic_model_nested_in_a_class():
            # #167: the model class is live and unmodified, yet its
            # checkpoint identity (module, __name__) does not import.
            # Every upstream failure path returns the raw dump dict, so
            # the miss is silent at read time. Fail at build instead.
            def it_raises_at_construction_naming_the_model():
                with pytest.raises(ValueError, match=r"_ModelHolder\.Cfg") as exc:
                    NamespaceAwareSerde(namespaces=(_NestedPayloadNs,))
                msg = str(exc.value)
                assert "_NestedPayloadNs.Placed.cfg" in msg
                assert "module scope" in msg

            def it_finds_the_model_inside_a_generic_annotation():
                with pytest.raises(ValueError, match=r"_ModelHolder\.Cfg"):
                    NamespaceAwareSerde(namespaces=(_NestedPayloadInGenericNs,))

        def when_an_event_field_is_a_module_level_pydantic_model():
            def it_constructs():
                NamespaceAwareSerde(namespaces=(_ModulePayloadNs,))

        def when_a_nested_model_is_only_a_class_var():
            def it_constructs():
                NamespaceAwareSerde(namespaces=(_ClassVarPayloadNs,))

        def when_an_event_field_is_a_pydantic_v1_model_nested_in_a_class():
            def it_raises_at_construction():
                with pytest.raises(ValueError, match=r"_ModelHolderV1\.Cfg"):
                    NamespaceAwareSerde(namespaces=(_NestedV1PayloadNs,))

        def when_a_sibling_forward_reference_sits_beside_a_nested_model():
            def it_still_finds_the_nested_model():
                with pytest.raises(ValueError, match=r"_ModelHolder\.Cfg"):
                    NamespaceAwareSerde(namespaces=(_SiblingRefPayloadNs,))

        def when_constructor_allowlist_explicitly_admits_a_type():
            # Asymmetric guard that actually proves delegation: with strict
            # mode plus an explicit allowlist, a class that *is* on the
            # allowlist must revive as itself, while a class that *isn't*
            # must demote to ``dict``. A subclass that hardcoded its own
            # permissive or strict path (i.e. didn't actually consult the
            # parent's allowlist) would fail one of these two halves —
            # which is the contract #68 broke.
            def it_revives_allowlisted_types_and_demotes_others():
                serde = NamespaceAwareSerde(
                    allowed_msgpack_modules=[
                        (PlainPayload.__module__, PlainPayload.__name__),
                    ],
                )
                allowed = PlainPayload(name="ok", count=1)
                blocked = PydanticPayload(name="no", count=2)

                allowed_back = serde.loads_typed(serde.dumps_typed(allowed))
                blocked_back = serde.loads_typed(serde.dumps_typed(blocked))

                assert isinstance(allowed_back, PlainPayload)
                assert allowed_back.name == "ok"
                assert allowed_back.count == 1
                # PydanticPayload is not in the allowlist, so the parent's
                # strict path demotes it. If the subclass were ignoring the
                # allowlist (the #68 regression), this would revive instead.
                assert isinstance(blocked_back, dict)
                assert blocked_back == {"name": "no", "count": 2}

    def describe_strict_msgpack_behavior():
        # Pins the actually-observable behavior of upstream's strict mode.
        # Documented in ``docs/event-migrations.md`` — if this test goes
        # red because LangGraph flips to raising, the docs need to update
        # in lockstep.
        #
        # ``LANGGRAPH_STRICT_MSGPACK=true`` resolves internally to
        # ``allowed_msgpack_modules=None`` at construction. The env var is
        # read at module import (cached on a module-level constant), so
        # monkeypatch.setenv can't flip it inside a test. Use the
        # equivalent constructor kwarg — same code path, fewer footguns.

        def when_strict_mode_blocks_a_non_allowlisted_class():
            def it_demotes_the_revival_to_a_dict():
                # Strict mode does NOT raise — it demotes the value to its
                # raw kwargs dict. Loud failure happens at the FIRST
                # consumer access (``revived.attr`` → AttributeError), not
                # at the serde boundary.
                serde = NamespaceAwareSerde(allowed_msgpack_modules=None)

                revived = serde.loads_typed(
                    serde.dumps_typed(PlainPayload(name="x", count=1))
                )

                assert isinstance(revived, dict)
                assert revived == {"name": "x", "count": 1}

    def describe_migrations():
        def when_an_event_log_holds_legacy_payloads_from_a_scatter_union():
            def it_revives_each_branch_independently():
                # A handler annotated ``Scatter[A | B]`` emits both branches
                # under different qualnames. Once renamed, a checkpoint that
                # captured pre-rename state holds a HETEROGENEOUS list of
                # legacy payloads. Each branch must migrate independently.

                migrations = [
                    _persona_rename("rename-persona-approved"),
                    _rename_migration(
                        "rename-story-approved",
                        old_module="legacy.story",
                        old_qualname="Legacy.StoryApproved",
                        new_module=Story.__module__,
                        new_qualname="Story.Approve.Approved",
                    ),
                ]
                serde = NamespaceAwareSerde(migrations=migrations)

                # Build a list payload: ``[legacy_persona, legacy_story]`` —
                # the shape an EventLog channel ends up with after a
                # ``Scatter[Union[A, B]]`` handler runs.
                p_ext = ormsgpack.Ext(
                    EXT_NAMESPACE_AWARE_EVENT,
                    ormsgpack.packb(
                        ("legacy.persona", "Legacy.Approved", {"note": "p"}),
                        option=_option,
                    ),
                )
                s_ext = ormsgpack.Ext(
                    EXT_NAMESPACE_AWARE_EVENT,
                    ormsgpack.packb(
                        ("legacy.story", "Legacy.StoryApproved", {"note": "s"}),
                        option=_option,
                    ),
                )
                payload = (
                    "msgpack",
                    ormsgpack.packb([p_ext, s_ext], option=_option),
                )

                revived = serde.loads_typed(payload)

                assert isinstance(revived, list)
                assert len(revived) == 2
                assert isinstance(revived[0], Persona.Approve.Approved)
                assert isinstance(revived[1], Story.Approve.Approved)
                assert revived[0].note == "p"
                assert revived[1].note == "s"

        def when_renamed_event_is_nested_inside_a_langgraph_interrupt():
            def it_revives_via_recursive_ext_hook():
                # HITL flows checkpoint the interrupted value wrapped in
                # ``Interrupt(value=..., id=...)``. The serde's EXT_INTERRUPT
                # branch already recurses with the same hook, so a renamed
                # event one layer down should rewrite through the same
                # rename table with zero extra code.

                migrations = [
                    _persona_rename("rename-legacy-persona"),
                ]
                serde = NamespaceAwareSerde(migrations=migrations)
                payload = _legacy_interrupted_payload(
                    module="legacy.persona",
                    qualname="Legacy.Approved",
                    kwargs={"note": "hitl"},
                    interrupt_id="i-1",
                )

                revived = serde.loads_typed(payload)

                assert isinstance(revived, Interrupt)
                assert revived.id == "i-1"
                assert isinstance(revived.value, Persona.Approve.Approved)
                assert revived.value.note == "hitl"

        def when_legacy_write_is_enabled():
            def it_encodes_under_the_oldest_known_qualname():
                # During a rolling deploy, new pods must write payloads that
                # old pods (running the previous release without the new
                # class definition) can still revive. ``legacy_write=True``
                # encodes under the OLDEST entry in the class's migration
                # history. After rollout completes, the user flips the flag
                # off in a follow-up release.
                serde = NamespaceAwareSerde(namespaces=[DecoReorg], legacy_write=True)
                event = DecoReorg.Persist.Persisted(note="x")

                kind, payload = serde.dumps_typed(event)

                # Old qualname appears in the encoded bytes; new qualname
                # does not. (Bytes contain msgpack-encoded strings, so the
                # raw byte search is sufficient — these qualnames don't
                # appear elsewhere in the encoded structure.)
                assert kind == "msgpack"
                assert b"DecoReorg.Persisted" in payload
                assert b"DecoReorg.Persist.Persisted" not in payload

            def it_round_trips_through_a_migration_aware_serde():
                # The legacy-written payload should revive under the live
                # class when read through a serde that carries the same
                # migration history (typically: the same code, same
                # ``namespaces=`` list, in the next pod).
                writer = NamespaceAwareSerde(namespaces=[DecoReorg], legacy_write=True)
                reader = NamespaceAwareSerde(namespaces=[DecoReorg])
                event = DecoReorg.Persist.Persisted(note="rolling")

                revived = reader.loads_typed(writer.dumps_typed(event))

                assert isinstance(revived, DecoReorg.Persist.Persisted)
                assert revived.note == "rolling"

        def when_decorator_records_the_old_qualname():
            def it_revives_legacy_payloads_at_the_decorated_class():
                # Authoring via the decorator should be equivalent to writing
                # the Migration list by hand — exercise the end-to-end
                # decorator path. Caller scopes collection to the relevant
                # namespace via ``namespaces=``.
                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                payload = synthesize_legacy_payload(
                    DecoReorg.__module__,
                    "DecoReorg.Persisted",
                    {"note": "from-legacy"},
                )

                revived = serde.loads_typed(payload)

                assert isinstance(revived, DecoReorg.Persist.Persisted)
                assert revived.note == "from-legacy"

        def when_a_subclass_inherits_a_decorated_parent():
            # ``@migrate_from`` metadata must not leak through MRO. A plain
            # subclass that doesn't redeclare its own history would
            # otherwise see ``__lge_migrate_from__`` via normal attribute
            # lookup — corrupting both the auto-collection path (duplicate
            # Migration entries → "Duplicate rename source" at construction)
            # and legacy_write (writes Child instances under the parent's
            # historic qualname).

            def it_does_not_emit_a_migration_for_the_subclass():
                # Build a serde scoped to SubLeak and inspect its rename
                # table: only Parent's historic identity should appear as
                # a source. Before the fix, Child inherited Parent's
                # marker and produced a second entry, triggering
                # "Duplicate rename source" at serde construction.
                serde = NamespaceAwareSerde(namespaces=[SubLeak])
                sources = [old_qn for (_, old_qn) in serde._rename_table]
                assert sources == ["SubLeak.OldParent"]

            def it_encodes_subclass_instances_under_their_own_qualname():
                # ``legacy_write=True`` consults the serde's oldest-historic
                # map (built during the namespace walk) instead of the
                # class's ``__lge_migrate_from__`` attribute directly. The
                # subclass has none of its own decorator metadata, so
                # legacy_write falls through to the live qualname rather
                # than borrowing the parent's historic name.
                serde = NamespaceAwareSerde(namespaces=[SubLeak], legacy_write=True)
                _, payload = serde.dumps_typed(SubLeak.Child(note="c", extra="x"))

                assert b"SubLeak.Child" in payload
                assert b"SubLeak.OldParent" not in payload

        def when_decorator_stacks_via_repeated_application():
            # Python applies decorators bottom-up. The bottom-most decorator
            # runs first, so its qualname is the OLDEST in the chain. The
            # stacking convention must match the multi-arg form
            # ``@migrate_from("A", "B")`` where A is the oldest.

            def it_treats_bottom_decorator_as_oldest():
                history = DecoStack.Persist.Persisted.__lge_migrate_from__
                assert [qn for _, qn in history] == [
                    "DecoStack.Oldest",
                    "DecoStack.Newer",
                ]

            def it_legacy_writes_under_the_oldest_qualname():
                # Real-world consequence: ``legacy_write=True`` reads
                # history[0] to pick the qualname for old-pod compatibility.
                # If the order is reversed, new pods emit bytes that no
                # previous release ever knew about — old pods fail to revive.
                serde = NamespaceAwareSerde(namespaces=[DecoStack], legacy_write=True)
                _, payload = serde.dumps_typed(DecoStack.Persist.Persisted(note="x"))

                assert b"DecoStack.Oldest" in payload
                assert b"DecoStack.Newer" not in payload
                assert b"DecoStack.Persist.Persisted" not in payload

        def when_decorator_stacks_a_multi_step_chain():
            def it_revives_payloads_from_any_historic_step():
                # ``@migrate_from("A", "B")`` declares the class lived at
                # ``A`` then ``B`` then its current location (oldest first).
                # Payloads from ANY historic step should land on the current
                # class after flatten.
                serde = NamespaceAwareSerde(namespaces=[DecoMulti])
                p_oldest = synthesize_legacy_payload(
                    DecoMulti.__module__, "DecoMulti.Persisted", {"note": "a"}
                )
                p_middle = synthesize_legacy_payload(
                    DecoMulti.__module__,
                    "DecoMulti.OldNest.Persisted",
                    {"note": "b"},
                )

                assert isinstance(
                    serde.loads_typed(p_oldest), DecoMulti.Persist.Persisted
                )
                assert isinstance(
                    serde.loads_typed(p_middle), DecoMulti.Persist.Persisted
                )

        def when_chained_renames_span_multiple_releases():
            def it_collapses_the_chain_to_the_final_class():
                # Simulate two refactors landed in successive releases:
                # release N renamed Legacy.A → Legacy.B; release N+1 renamed
                # Legacy.B → Persona.Approve.Approved. Payloads from BOTH
                # historic identities should land on the current class.

                migrations = [
                    _rename_migration(
                        "release-N",
                        old_module="legacy.persona",
                        old_qualname="Legacy.A",
                        new_module="legacy.persona",
                        new_qualname="Legacy.B",
                    ),
                    _rename_migration(
                        "release-N+1",
                        old_module="legacy.persona",
                        old_qualname="Legacy.B",
                        new_module=Persona.__module__,
                        new_qualname="Persona.Approve.Approved",
                    ),
                ]

                serde = NamespaceAwareSerde(migrations=migrations)
                payload_a = synthesize_legacy_payload(
                    "legacy.persona", "Legacy.A", {"note": "from-A"}
                )
                payload_b = synthesize_legacy_payload(
                    "legacy.persona", "Legacy.B", {"note": "from-B"}
                )

                revived_a = serde.loads_typed(payload_a)
                revived_b = serde.loads_typed(payload_b)

                assert isinstance(revived_a, Persona.Approve.Approved)
                assert isinstance(revived_b, Persona.Approve.Approved)
                assert revived_a.note == "from-A"
                assert revived_b.note == "from-B"

        def when_a_required_field_was_added():
            # Shared migration list for both sub-cases below.

            def when_old_payload_omits_the_field():
                def it_injects_the_default_value():
                    # Without AddField the dataclass constructor raises
                    # TypeError because ``command_id`` is required.
                    serde = NamespaceAwareSerde(
                        migrations=[
                            _persisted_field_migration(
                                "add-command-id", "command_id", default="legacy"
                            )
                        ]
                    )

                    revived = serde.loads_typed(_persisted_payload(note="old"))

                    assert isinstance(revived, NestedPersist.Persist.Persisted)
                    assert revived.command_id == "legacy"
                    assert revived.note == "old"

            def when_default_factory_is_used_for_mutable_value():
                def it_invokes_the_factory_per_payload():
                    # Two distinct payloads must NOT share the same list
                    # object — that would be a classic mutable-default bug
                    # made worse by living one layer down inside the serde.
                    #
                    # Override the default tuple field with a list-typed one
                    # by going through kwargs directly — we want to observe
                    # whether revival hands back the SAME list instance.
                    serde = NamespaceAwareSerde(
                        migrations=[
                            _persisted_field_migration(
                                "add-tags", "tags", default_factory=list
                            )
                        ]
                    )

                    r1 = serde.loads_typed(
                        _persisted_payload(command_id="c1", note="a")
                    )
                    r2 = serde.loads_typed(
                        _persisted_payload(command_id="c2", note="b")
                    )

                    # Frozen dataclass coerces our list into the declared
                    # ``tuple[str, ...]`` only when we pass a tuple — we
                    # passed nothing, so the AddField factory list goes in
                    # raw. Identity comparison surfaces aliasing.
                    assert r1.tags is not r2.tags

            def when_old_payload_already_carries_the_field():
                def it_preserves_the_existing_value():
                    # If a payload predates the AddField migration but
                    # happens to carry the field (e.g. a backfill ran),
                    # setdefault must not overwrite the existing value.
                    serde = NamespaceAwareSerde(
                        migrations=[
                            _persisted_field_migration(
                                "add-command-id", "command_id", default="legacy"
                            )
                        ]
                    )

                    revived = serde.loads_typed(
                        _persisted_payload(command_id="from-old", note="n")
                    )

                    assert revived.command_id == "from-old"

        def when_event_was_renamed():
            def it_revives_under_new_qualname():
                # Simulate a checkpoint produced by an older build where the
                # event class lived at "legacy.module:Legacy.Approved", later
                # renamed to the live `Persona.Approve.Approved`.

                payload = synthesize_legacy_payload(
                    module="legacy.persona",
                    qualname="Legacy.Approved",
                    kwargs={"note": "hi"},
                )
                migrations = [
                    _persona_rename("rename-legacy-persona-approved"),
                ]

                serde = NamespaceAwareSerde(migrations=migrations)
                revived = serde.loads_typed(payload)

                assert isinstance(revived, Persona.Approve.Approved)
                assert revived.note == "hi"

        def when_event_was_renamed_and_gained_a_required_field():
            # The realistic two-release scenario: between checkpoint write
            # and current read, the event was both relocated under a Command
            # AND grew a new required field. AddField is keyed on the
            # POST-rename identity, so the two ops must compose — rename
            # first, then field injection on the rewritten identity.
            def it_applies_rename_then_addfield_in_a_single_revival():
                from langgraph_events.serde.migrations import (
                    AddField,
                    Migration,
                    RenameEvent,
                )

                payload = synthesize_legacy_payload(
                    module="legacy.persist",
                    qualname="Legacy.Persisted",
                    # Old payload: pre-rename identity, no command_id.
                    kwargs={"note": "old"},
                )
                migrations = [
                    Migration(
                        name="rename-and-add-command-id",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persist",
                                old_qualname="Legacy.Persisted",
                                new_module=NestedPersist.__module__,
                                new_qualname="NestedPersist.Persist.Persisted",
                            ),
                            AddField(
                                module=NestedPersist.__module__,
                                qualname="NestedPersist.Persist.Persisted",
                                field="command_id",
                                default="legacy",
                            ),
                        ),
                    ),
                ]

                serde = NamespaceAwareSerde(migrations=migrations)
                revived = serde.loads_typed(payload)

                assert isinstance(revived, NestedPersist.Persist.Persisted)
                assert revived.command_id == "legacy"
                assert revived.note == "old"

    def describe_Migration_name():
        # ``name`` labels a migration for use in diagnostics. Hand-authored
        # migrations don't always have a sensible label to offer, and the
        # serde validates fine without one — so the field is optional.

        def when_name_is_omitted():
            def it_defaults_to_an_empty_string():
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                m = Migration(
                    operations=(
                        RenameEvent(
                            old_module="legacy.persona",
                            old_qualname="Legacy.Approved",
                            new_module=Persona.__module__,
                            new_qualname="Persona.Approve.Approved",
                        ),
                    )
                )
                # Default is empty; users who care provide a label.
                assert m.name == ""

            def it_can_be_passed_directly_to_NamespaceAwareSerde():
                # Validates that nothing in the flatten path treats a blank
                # name as a sentinel that needs replacing or guarding.
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                NamespaceAwareSerde(
                    migrations=[
                        Migration(
                            operations=(
                                RenameEvent(
                                    old_module="legacy.persona",
                                    old_qualname="Legacy.Approved",
                                    new_module=Persona.__module__,
                                    new_qualname="Persona.Approve.Approved",
                                ),
                            )
                        ),
                    ]
                )

    def describe_Migration_sugar():
        # ``Migration.rename`` and ``Migration.add_field`` are single-op
        # builders. Both are kw-only and their parameter names must align with
        # the underlying op dataclasses so users can move between sugar and
        # raw form without renaming arguments.

        def when_Migration_rename_builds_a_single_RenameEvent():
            def it_matches_the_explicit_form():
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                sugar = Migration.rename(
                    name="r",
                    old_module="legacy.persona",
                    old_qualname="Legacy.Approved",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                )
                explicit = Migration(
                    name="r",
                    operations=(
                        RenameEvent(
                            old_module="legacy.persona",
                            old_qualname="Legacy.Approved",
                            new_module=Persona.__module__,
                            new_qualname="Persona.Approve.Approved",
                        ),
                    ),
                )
                assert sugar == explicit

        def when_Migration_add_field_builds_a_single_AddField():
            # ``name`` consistently refers to a migration label; ``field``
            # consistently refers to the dataclass field being added. The
            # sugar and the raw form must use the same convention so users
            # don't have to translate kwargs when moving between them.

            def it_matches_the_explicit_form():
                from langgraph_events.serde.migrations import (
                    AddField,
                    Migration,
                )

                sugar = Migration.add_field(
                    "add-tags",
                    module=NestedPersist.__module__,
                    qualname="NestedPersist.Persist.Persisted",
                    field="tags",
                    default_factory=list,
                )
                explicit = Migration(
                    name="add-tags",
                    operations=(
                        AddField(
                            module=NestedPersist.__module__,
                            qualname="NestedPersist.Persist.Persisted",
                            field="tags",
                            default_factory=list,
                        ),
                    ),
                )
                assert sugar == explicit

            def it_rejects_the_old_name_kw_on_AddField():
                # ``AddField(name=...)`` was the previous spelling for the
                # dataclass field name — same word as ``Migration.name``
                # (the migration label) which made the two indistinguishable
                # at call sites. ``name`` is now reserved for migration
                # labels everywhere; the dataclass field name is ``field``.
                from langgraph_events.serde.migrations import AddField

                with pytest.raises(TypeError):
                    AddField(  # type: ignore[call-arg]
                        module=NestedPersist.__module__,
                        qualname="NestedPersist.Persist.Persisted",
                        name="tags",  # ← old kw — must not silently accept
                        default_factory=list,
                    )

        def when_the_live_target_is_passed_as_a_class():
            # The new (post-rename) / AddField target identity always names
            # a class that exists in code, so passing the class itself is
            # refactor-safe — an IDE rename moves with it instead of
            # silently breaking a string. Historic identities stay strings
            # (the old class is gone).

            def it_derives_new_identity_on_Migration_rename():
                from langgraph_events.serde.migrations import Migration

                via_class = Migration.rename(
                    name="r",
                    old_module="legacy.persona",
                    old_qualname="Legacy.Approved",
                    to=Persona.Approve.Approved,
                )
                via_strings = Migration.rename(
                    name="r",
                    old_module="legacy.persona",
                    old_qualname="Legacy.Approved",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                )
                assert via_class == via_strings

            def it_derives_identity_on_Migration_add_field():
                from langgraph_events.serde.migrations import Migration

                via_class = Migration.add_field(
                    "add-tags",
                    target=NestedPersist.Persist.Persisted,
                    field="tags",
                    default_factory=list,
                )
                via_strings = Migration.add_field(
                    "add-tags",
                    module=NestedPersist.__module__,
                    qualname="NestedPersist.Persist.Persisted",
                    field="tags",
                    default_factory=list,
                )
                assert via_class == via_strings

            def it_round_trips_a_legacy_payload_via_the_class_form():
                from langgraph_events.serde.migrations import Migration

                serde = NamespaceAwareSerde(
                    migrations=[
                        Migration.rename(
                            old_module="legacy.persona",
                            old_qualname="Legacy.Approved",
                            to=Persona.Approve.Approved,
                        )
                    ]
                )
                revived = serde.loads_typed(
                    synthesize_legacy_payload(
                        "legacy.persona", "Legacy.Approved", {"note": "n"}
                    )
                )
                assert isinstance(revived, Persona.Approve.Approved)
                assert revived.note == "n"

        def when_both_the_class_and_string_identity_are_given():
            def it_rejects_the_ambiguous_call_on_rename():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError, match="to"):
                    Migration.rename(
                        old_module="legacy.persona",
                        old_qualname="Legacy.Approved",
                        to=Persona.Approve.Approved,
                        new_qualname="Persona.Approve.Approved",
                    )

            def it_rejects_the_ambiguous_call_on_add_field():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError, match="target"):
                    Migration.add_field(
                        target=NestedPersist.Persist.Persisted,
                        module=NestedPersist.__module__,
                        field="tags",
                        default_factory=list,
                    )

        def when_neither_class_nor_string_identity_is_given():
            def it_rejects_the_underspecified_rename():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError):
                    Migration.rename(
                        old_module="legacy.persona",
                        old_qualname="Legacy.Approved",
                    )

        def when_name_is_omitted_on_sugar():
            # ``name`` is documented as optional everywhere. The bare
            # ``Migration(operations=(...))`` form already defaults it to
            # ``""``; the sugar classmethods must match so the docs claim
            # holds.

            def it_defaults_name_on_Migration_rename():
                from langgraph_events.serde.migrations import Migration

                m = Migration.rename(
                    old_module="legacy.persona",
                    old_qualname="Legacy.Approved",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                )
                assert m.name == ""

            def it_defaults_name_on_Migration_add_field():
                from langgraph_events.serde.migrations import Migration

                m = Migration.add_field(
                    module=NestedPersist.__module__,
                    qualname="NestedPersist.Persist.Persisted",
                    field="tags",
                    default_factory=list,
                )
                assert m.name == ""

        def when_target_identity_is_under_or_over_specified():
            # Characterizes the exact diagnostics ``Migration.rename`` and
            # ``Migration.add_field`` raise. Both classmethods run the same
            # "exactly one of class-object vs module/qualname" check; these
            # asserts pin the messages so the shared helper can be factored
            # out without silently changing user-facing text.

            def it_states_the_exact_rename_both_supplied_message():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError) as exc:
                    Migration.rename(
                        old_module="legacy.persona",
                        old_qualname="Legacy.Approved",
                        to=Persona.Approve.Approved,
                        new_qualname="Persona.Approve.Approved",
                    )
                assert str(exc.value) == (
                    "Migration.rename: pass either `to=<class>` or "
                    "`new_module`/`new_qualname`, not both."
                )

            def it_states_the_exact_rename_neither_supplied_message():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError) as exc:
                    Migration.rename(
                        old_module="legacy.persona",
                        old_qualname="Legacy.Approved",
                    )
                assert str(exc.value) == (
                    "Migration.rename: provide the live target as "
                    "`to=<class>` or both `new_module` and `new_qualname`."
                )

            def it_states_the_exact_add_field_both_supplied_message():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError) as exc:
                    Migration.add_field(
                        target=NestedPersist.Persist.Persisted,
                        module=NestedPersist.__module__,
                        field="tags",
                        default_factory=list,
                    )
                assert str(exc.value) == (
                    "Migration.add_field: pass either `target=<class>` or "
                    "`module`/`qualname`, not both."
                )

            def it_states_the_exact_add_field_neither_supplied_message():
                from langgraph_events.serde.migrations import Migration

                with pytest.raises(ValueError) as exc:
                    Migration.add_field(field="tags", default_factory=list)
                assert str(exc.value) == (
                    "Migration.add_field: provide the target as "
                    "`target=<class>` or both `module` and `qualname`."
                )

    def describe_migrations_validation():
        # Validation runs once at serde construction. Every error here would
        # otherwise surface as a ``ValueError`` on first production read,
        # which is the worst possible time to discover it.

        def when_chain_target_does_not_resolve():
            def it_raises_pointing_at_the_dead_target():
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="typo",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                # `Approvedd` does not exist on Persona.Approve.
                                new_qualname="Persona.Approve.Approvedd",
                            ),
                        ),
                    ),
                ]

                with pytest.raises(ValueError, match=r"Approvedd|does not resolve"):
                    NamespaceAwareSerde(migrations=migrations)

        def when_duplicate_old_source():
            def it_raises_naming_the_ambiguous_source():

                migrations = [
                    _persona_rename("rename-to-persona"),
                    _rename_migration(
                        "rename-to-story",
                        old_module="legacy.persona",
                        old_qualname="Legacy.Approved",
                        new_module=Story.__module__,
                        new_qualname="Story.Approve.Approved",
                    ),
                ]

                # Both migration names must appear so the user can grep the
                # culprits — otherwise the diagnostic lists only the rename
                # targets, which is harder to find in a large migrations list.
                with pytest.raises(ValueError) as exc:
                    NamespaceAwareSerde(migrations=migrations)
                assert "Duplicate rename source" in str(exc.value)
                assert "rename-to-persona" in str(exc.value)
                assert "rename-to-story" in str(exc.value)

        def when_old_qualname_still_resolves():
            def it_raises_to_prevent_shadowing_the_live_class():

                # `Persona.Approve.Approved` IS live — declaring a migration
                # whose old name points at it would silently rewrite live
                # payloads on read.
                migrations = [
                    _rename_migration(
                        "shadows-live",
                        old_module=Persona.__module__,
                        old_qualname="Persona.Approve.Approved",
                        new_module=Story.__module__,
                        new_qualname="Story.Approve.Approved",
                    ),
                ]

                with pytest.raises(ValueError, match=r"resolves to a currently-live"):
                    NamespaceAwareSerde(migrations=migrations)

        def when_addfield_target_is_unresolvable():
            def it_raises_naming_the_dead_target():
                from langgraph_events.serde.migrations import (
                    AddField,
                    Migration,
                )

                migrations = [
                    Migration(
                        name="add-to-ghost",
                        operations=(
                            AddField(
                                module="legacy.persona",
                                qualname="Legacy.Ghost",
                                field="x",
                                default=1,
                            ),
                        ),
                    ),
                ]

                with pytest.raises(ValueError, match=r"neither resolves"):
                    NamespaceAwareSerde(migrations=migrations)

        def when_addfield_default_is_a_mutable_scalar():
            def it_raises_steering_user_to_default_factory():
                from langgraph_events.serde.migrations import AddField

                with pytest.raises(ValueError, match=r"default_factory=list"):
                    AddField(
                        module=NestedPersist.__module__,
                        qualname="NestedPersist.Persist.Persisted",
                        field="tags",
                        default=[],
                    )

        def when_addfield_specifies_both_default_and_factory():
            def it_raises_requiring_exactly_one():
                from langgraph_events.serde.migrations import AddField

                with pytest.raises(ValueError, match=r"exactly one"):
                    AddField(
                        module=NestedPersist.__module__,
                        qualname="NestedPersist.Persist.Persisted",
                        field="tags",
                        default=(),
                        default_factory=tuple,
                    )

        def when_operation_is_not_a_known_type():
            # ``Migration.operations`` is typed as ``RenameEvent | AddField``,
            # but the validator iterated with ``if/elif`` and no ``else``, so a
            # stray object dropped into ``operations`` was silently ignored.
            # That hides authoring errors — surface them at construction.

            def it_raises_naming_the_unknown_type():
                from langgraph_events.serde.migrations import Migration

                class _Bogus:
                    pass

                with pytest.raises(TypeError, match=r"_Bogus"):
                    NamespaceAwareSerde(
                        migrations=[
                            Migration(operations=(_Bogus(),), name="bogus"),  # type: ignore[arg-type]
                        ]
                    )

        def when_cycle_exists_in_chain():
            def it_raises_naming_the_cycle_start():

                migrations = [
                    _rename_migration(
                        "forward",
                        old_module="legacy.a",
                        old_qualname="A",
                        new_module="legacy.b",
                        new_qualname="B",
                    ),
                    _rename_migration(
                        "reverse",
                        old_module="legacy.b",
                        old_qualname="B",
                        new_module="legacy.a",
                        new_qualname="A",
                    ),
                ]

                with pytest.raises(ValueError, match=r"Cycle"):
                    NamespaceAwareSerde(migrations=migrations)

        def when_old_qualname_equals_new_qualname():
            def it_raises_naming_the_self_loop():
                # A self-loop is the trivial cycle: ``A → A``. Earlier
                # versions surfaced this as "Cycle in migration chain"
                # which is technically true but misleads — the actual
                # cause is that the user wrote a rename whose target
                # equals its source (often a typo or stale paste).

                migrations = [
                    _rename_migration(
                        "self",
                        old_module=Persona.__module__,
                        old_qualname="Persona.Approve.Approved",
                        new_module=Persona.__module__,
                        new_qualname="Persona.Approve.Approved",
                    ),
                ]

                with pytest.raises(ValueError, match=r"self-loop|maps to itself"):
                    NamespaceAwareSerde(migrations=migrations)

    def describe_scoped_collection():
        # The serde's decorator-driven migration collection is scoped to
        # namespaces explicitly passed via ``namespaces=``. Imported but
        # unrelated namespaces (e.g. fixtures from another module loaded
        # by the test runner) must not contribute migrations or trigger
        # ``legacy_write`` relabelling.

        def when_a_namespace_is_passed_to_namespaces_kwarg():
            def it_collects_decorators_only_from_those_namespaces():
                # ``ScopeFixtureA`` carries ``@migrate_from``; scoping to
                # ``[ScopeFixtureB]`` must NOT pick up A's migration.
                serde = NamespaceAwareSerde(namespaces=[ScopeFixtureB])

                a_sources = [
                    old_qn
                    for (_, old_qn) in serde._rename_table
                    if old_qn.startswith("ScopeFixtureA.")
                ]
                assert a_sources == []

            def it_collects_decorators_inside_the_passed_namespaces():
                # Sanity check the positive case: scoping to ``[A]`` DOES
                # collect A's decorator-driven migration.
                serde = NamespaceAwareSerde(namespaces=[ScopeFixtureA])

                a_sources = [
                    old_qn
                    for (_, old_qn) in serde._rename_table
                    if old_qn.startswith("ScopeFixtureA.")
                ]
                assert a_sources == ["ScopeFixtureA.Persisted"]

        def when_no_namespaces_are_passed():
            def it_collects_no_decorator_driven_migrations():
                # Empty default = no decorator collection. Only hand-
                # authored ``migrations=`` entries (none here) would apply.
                serde = NamespaceAwareSerde()

                assert serde._rename_table == {}
                assert serde._addfield_table == {}

        def when_class_is_decorated_but_out_of_scope():
            def with_legacy_write_enabled():
                def it_encodes_under_current_qualname():
                    # Encode/decode symmetry: ``legacy_write=True`` must NOT
                    # relabel a decorated class whose migration is not in
                    # the serde's read-side rename table. Otherwise bytes
                    # go out under a historic name the reader can't
                    # migrate.
                    serde = NamespaceAwareSerde(
                        namespaces=[ScopeFixtureB],
                        legacy_write=True,
                    )

                    _, payload = serde.dumps_typed(
                        ScopeFixtureA.Persist.Persisted(note="x")
                    )

                    assert b"ScopeFixtureA.Persist.Persisted" in payload
                    assert b"ScopeFixtureA.Persisted" not in payload

    def describe_scope_first_identity_resolution():
        # Revival consults the serde's own ``namespaces=`` scope before it
        # falls back to importing the module and walking the qualname. The
        # scope is the authority on what this serde can revive; the import
        # walk is the fallback for identities the walk never reached
        # (module-level IntegrationEvents, framework SystemEvents). See #150.

        def when_the_namespace_is_defined_inside_a_function():
            def it_revives_the_event():
                trading = _local_trading()
                serde = NamespaceAwareSerde(namespaces=[trading])

                revived = serde.loads_typed(
                    serde.dumps_typed(trading.Place.Placed(sym="AAPL"))
                )

                assert type(revived) is trading.Place.Placed
                assert revived.sym == "AAPL"

        def when_a_scoped_class_carries_migrate_from():
            def it_revives_the_historic_payload():
                # Migration validation runs at construction and asks the
                # same question the read path asks — so it, too, must see
                # the scope, or a rename chain that terminates on a
                # function-local class reads as a dead end.
                trading = _local_migrated_trading()
                serde = NamespaceAwareSerde(namespaces=[trading])

                revived = serde.loads_typed(
                    synthesize_legacy_payload(
                        __name__, "Trading.Place.Filled", {"sym": "AAPL"}
                    )
                )

                assert type(revived) is trading.Place.Placed
                assert revived.sym == "AAPL"

        def when_a_scoped_class_carries_backfill():
            def it_injects_the_default_into_the_older_payload():
                # The AddField target lands in the post-rename bucket only
                # if it resolves — and its field is validated against the
                # live class. Both questions need the scope.
                trading = _local_backfilled_trading()
                serde = NamespaceAwareSerde(namespaces=[trading])

                revived = serde.loads_typed(
                    synthesize_legacy_payload(
                        trading.Place.Placed.__module__,
                        trading.Place.Placed.__qualname__,
                        {"sym": "AAPL"},
                    )
                )

                assert type(revived) is trading.Place.Placed
                assert revived.venue == "NYSE"

        def when_a_rename_source_is_still_live_in_scope():
            def it_raises_to_prevent_shadowing_the_live_class():
                # The rename runs before resolution, so it shadows the live
                # class whether or not that class is importable. An
                # import-only shadow check simply cannot see a
                # function-local one, and lets the rewrite through.
                trading = _local_trading()
                migrations = [
                    _rename_migration(
                        "shadows-scoped",
                        old_module=trading.Place.Placed.__module__,
                        old_qualname=trading.Place.Placed.__qualname__,
                        new_module=Persona.__module__,
                        new_qualname="Persona.Approve.Approved",
                    ),
                ]

                with pytest.raises(ValueError, match=r"resolves to a currently-live"):
                    NamespaceAwareSerde(namespaces=[trading], migrations=migrations)

    def describe_revivable_identities():
        # The set of ``(module, qualname)`` this serde will accept on a
        # read: live classes in scope, plus every historic identity a
        # rename migration can rewrite (decorator-driven AND hand-authored,
        # unioned — users want one question answered, "will this revive?",
        # not "live or migrated?").

        def it_equals_the_identities_the_namespace_walk_reaches():
            # Exact equality, not membership: the gate that consumes this
            # set (``assert_all_baselined_cover``) only asks whether a
            # baselined identity is present, so an identity the walk never
            # reached would make the gate pass on a checkpoint the read
            # path cannot actually revive (#154).
            module = WalkShapes.__module__

            ids = NamespaceAwareSerde(namespaces=[WalkShapes]).revivable_identities()

            assert ids == {
                # DomainEvent nested directly in the Namespace.
                (module, "WalkShapes.Recorded"),
                # Non-DomainEvent nested in the Namespace for locality.
                (module, "WalkShapes.Blocked"),
                # The Command classes themselves — they are Events too, and
                # a checkpointed command is as revivable as its outcomes.
                (module, "WalkShapes.Record"),
                (module, "WalkShapes.Amend"),
                # Nested inside a Command: reached only by the
                # ``recurse_commands=True`` descent.
                (module, "WalkShapes.Record.Stored"),
                (module, "WalkShapes.Record.Stalled"),
                (module, "WalkShapes.Amend.Amended"),
                (module, "WalkShapes.Amend.Rejected"),
            }
            # The namespace itself and its non-event members contribute
            # nothing — asserted by the equality above, called out here
            # because each is a distinct branch of the walk.
            #
            # The ``Outcomes`` aliases are NOT pinned here, and this set
            # cannot pin them: for single-outcome ``Record`` the alias is the
            # same class object as ``Stored``, which set semantics absorb,
            # and for multi-outcome ``Amend`` it is a ``UnionType`` that
            # fails the walk's ``isinstance(attr, type)`` guard on its own.
            # The ``OUTCOMES_ATTR`` skip earns its keep in the walk's *list*
            # (double-visited callbacks, double-applied AddFields), which
            # ``it_lists_the_outcome_only_once`` covers.
            assert (module, "WalkShapes") not in ids
            assert (module, "WalkShapes.Helper") not in ids

        def it_unions_the_walk_and_every_rename_source():
            # The other half of the set: historic identities are revivable
            # because a rename rewrites them, from either source of
            # migrations. Exact equality again — AddField targets and
            # rename *destinations* must add nothing of their own.
            #
            # The AddField is keyed on a live class OUTSIDE namespaces=, so
            # it lands in _addfield_table with no overlap against the walk or
            # the rename sources. Without it the claim about AddField targets
            # would be a comment rather than an assertion: an empty table
            # cannot distinguish "not unioned in" from "nothing to union".
            serde = NamespaceAwareSerde(
                namespaces=[DecoReorg],
                migrations=[
                    _persona_rename("legacy-rename"),
                    _add_field_migration(
                        "out-of-scope-fill",
                        module=Persona.__module__,
                        qualname="Persona.Approve.Approved",
                        field="note",
                        default="",
                    ),
                ],
            )

            ids = serde.revivable_identities()

            assert ids == {
                # Live classes reached by the namespace walk.
                (DecoReorg.__module__, "DecoReorg.Persist"),
                (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                # Decorator-driven historic source.
                (DecoReorg.__module__, "DecoReorg.Persisted"),
                # Hand-authored historic source.
                ("legacy.persona", "Legacy.Approved"),
            }

        def it_returns_a_read_only_frozenset():
            serde = NamespaceAwareSerde(namespaces=[DecoReorg])

            assert isinstance(serde.revivable_identities(), frozenset)

    def describe_synthesize_legacy_payload():
        # The public helper that lifts the wire-format byte assembly out of
        # tests/examples so users don't import ``_option`` /
        # ``EXT_NAMESPACE_AWARE_EVENT`` to prove "release N bytes still
        # revive under the current migration table".

        def it_round_trips_through_NamespaceAwareSerde_at_current_class():
            from langgraph_events.serde import synthesize_legacy_payload

            serde = NamespaceAwareSerde(namespaces=[DecoReorg])
            payload = synthesize_legacy_payload(
                DecoReorg.__module__,
                "DecoReorg.Persisted",
                {"note": "from-legacy"},
            )

            revived = serde.loads_typed(payload)

            assert isinstance(revived, DecoReorg.Persist.Persisted)
            assert revived.note == "from-legacy"

    def describe_assert_all_baselined_cover():
        # The CI gate: every identity the previous release wrote must be
        # either still live in this serde's namespaces or covered by a
        # rename migration, else the next production read of that payload
        # would fail. Weakest of the three gates — a set-membership check
        # that neither resolves nor constructs.

        def when_baseline_identity_has_no_migration():
            def it_raises_naming_the_uncovered_identity(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    MigrationCoverageError,
                    assert_all_baselined_cover,
                )

                serde = NamespaceAwareSerde()
                baseline = _baseline_file(tmp_path, ("ghost.mod", "Ghost.Gone"))

                with pytest.raises(MigrationCoverageError) as excinfo:
                    assert_all_baselined_cover(serde, baseline)

                assert "Ghost.Gone" in str(excinfo.value)
                assert excinfo.value.uncovered == (("ghost.mod", "Ghost.Gone"),)
                # Grammar must agree with the count: one identity, not "are".
                assert "1 identity in the baseline is neither" in str(excinfo.value)
                assert "add @migrate_from" in str(excinfo.value)

        def when_the_baseline_lists_an_unmigrated_retired_identity():
            def it_raises_naming_the_retired_remedy(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    MigrationCoverageError,
                    assert_all_baselined_cover,
                )

                serde = NamespaceAwareSerde(namespaces=[Retiring])
                baseline = _baseline_v3_file(
                    tmp_path, retired=(("ghost.mod", "Ghost.Gone", ["x"]),)
                )

                with pytest.raises(MigrationCoverageError) as excinfo:
                    assert_all_baselined_cover(serde, baseline)

                message = str(excinfo.value)
                assert excinfo.value.uncovered == (("ghost.mod", "Ghost.Gone"),)
                assert message.count("Ghost.Gone") == 1
                assert "regenerate" not in message
                assert "unrevivable_threads() == {}" in message
                assert "delete its `retired` entry" in message

        def when_unifying_the_gate_error_base():
            def it_is_an_assertion_error_subclass():
                # Unified error base: every gate raises AssertionError so
                # callers catch them uniformly, while cover keeps its
                # structured ``.uncovered`` tuple.
                from langgraph_events.serde.migrations import (
                    MigrationCoverageError,
                )

                assert issubclass(MigrationCoverageError, AssertionError)

        def when_baseline_identity_has_a_decorator_migration():
            def it_passes_silently(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_cover,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path, (DecoReorg.__module__, "DecoReorg.Persisted")
                )

                assert_all_baselined_cover(serde, baseline)

        def when_baseline_identity_is_still_live():
            def it_passes_silently(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_cover,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                )

                assert_all_baselined_cover(serde, baseline)

        def when_baseline_identity_has_a_hand_authored_migration():
            def it_passes_silently(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_cover,
                )

                serde = NamespaceAwareSerde(
                    migrations=[
                        _persona_rename("legacy-rename"),
                    ],
                )
                baseline = _baseline_file(
                    tmp_path, ("legacy.persona", "Legacy.Approved")
                )

                assert_all_baselined_cover(serde, baseline)

    def describe_assert_all_baselined_revive():
        # Stronger than assert_all_baselined_cover: instead of a set-membership
        # check, it pushes a synthesized legacy payload for every baselined
        # identity through the real ext-hook and asserts it revives to an
        # Event. Zero per-event maintenance — a new @migrate_from plus a
        # regenerated baseline is covered with no new test code.

        def when_every_baselined_identity_is_renamed_or_live():
            def it_revives_them_all_through_the_read_path(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persisted"),
                    (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                )

                assert_all_baselined_revive(serde, baseline)

        def when_an_init_false_field_has_no_default():
            def it_gives_that_field_no_placeholder(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DerivedNoDefault])
                baseline = _baseline_file(
                    tmp_path,
                    (DerivedNoDefault.__module__, "DerivedNoDefault.Placed"),
                )

                assert_all_baselined_revive(serde, baseline)

        def when_a_baselined_identity_is_neither_live_nor_migrated():
            def it_raises_naming_the_uncovered_identity(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persisted"),
                    ("ghost.mod", "Ghost.Gone"),
                )

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_revive(serde, baseline)

                assert "Ghost.Gone" in str(excinfo.value)
                assert "DecoReorg.Persisted" not in str(excinfo.value)
                assert "add @migrate_from" in str(excinfo.value)

        def when_a_baselined_identity_is_only_reachable_in_scope():
            def it_revives_through_the_scope(tmp_path: Any):
                # ``loads_typed`` already resolves scope-first, but the
                # required-field placeholders the gate synthesizes must be
                # read off the same class the reader will build — otherwise
                # a function-local ``Placed(sym=...)`` is constructed with
                # no kwargs and the gate reports a synthetic TypeError.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                trading = _local_trading()
                serde = NamespaceAwareSerde(namespaces=[trading])
                baseline = _baseline_file(
                    tmp_path,
                    (
                        trading.Place.Placed.__module__,
                        trading.Place.Placed.__qualname__,
                    ),
                )

                assert_all_baselined_revive(serde, baseline)

        def when_the_live_class_dropped_a_recorded_field():
            # The stored blob still carries the field. A v3 baseline records
            # it, so the gate sends it and the live class rejects it: the
            # failure a real checkpoint would raise.
            def it_fails_naming_the_recorded_field(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_v3_file(
                    tmp_path,
                    events=(
                        (
                            DecoReorg.__module__,
                            "DecoReorg.Persist.Persisted",
                            ["legacy_flag", "note"],
                        ),
                    ),
                )

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_revive(serde, baseline)

                message = str(excinfo.value)
                assert "DecoReorg.Persist.Persisted" in message
                assert "recorded 'legacy_flag'" in message
                # One remedy, from the read path: the gate adds the cause
                # sentence and does not repeat it.
                assert "drop it from the payload with @transform_fields" in message
                assert message.count("from the payload") == 1

            def with_a_baseline_that_predates_v3():
                def it_passes(tmp_path: Any):
                    # No field record, so nothing to send: the documented
                    # degrade until the project re-baselines.
                    from langgraph_events.serde.migrations import (
                        assert_all_baselined_revive,
                    )

                    serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                    baseline = _baseline_file(
                        tmp_path,
                        (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                    )

                    assert_all_baselined_revive(serde, baseline)

        def when_a_field_is_dropped_and_the_baseline_rewritten():
            def it_still_fails(tmp_path: Any):
                # The write keeps every field ever recorded, so a plain
                # re-baseline after the drop cannot blind the gate.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )
                from langgraph_events.serde.migrations.detect import write_baseline

                baseline = _baseline_v3_file(
                    tmp_path,
                    events=(
                        (
                            DecoReorg.__module__,
                            "DecoReorg.Persist.Persisted",
                            ["legacy_flag", "note"],
                        ),
                    ),
                )
                write_baseline(EventGraph.from_namespaces(DecoReorg), baseline)
                serde = NamespaceAwareSerde(namespaces=[DecoReorg])

                with pytest.raises(AssertionError, match="legacy_flag"):
                    assert_all_baselined_revive(serde, baseline)

        def when_a_defaulted_field_is_validated_in_post_init():
            def it_passes_after_a_v3_write(tmp_path: Any):
                # The record names ``name``. The live class still accepts
                # it, so no ``None`` placeholder reaches the validator.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )
                from langgraph_events.serde.migrations.detect import write_baseline

                baseline = tmp_path / "baseline.json"
                write_baseline(EventGraph.from_namespaces(ValidatedDefault), baseline)
                serde = NamespaceAwareSerde(namespaces=[ValidatedDefault])

                assert_all_baselined_revive(serde, baseline)

        def when_the_live_class_has_an_init_false_field():
            def it_stays_green_after_a_v3_write(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )
                from langgraph_events.serde.migrations.detect import write_baseline

                baseline = tmp_path / "baseline.json"
                write_baseline(EventGraph.from_namespaces(Derived), baseline)
                serde = NamespaceAwareSerde(namespaces=[Derived])

                assert_all_baselined_revive(serde, baseline)

        def when_the_live_class_gained_a_backfilled_required_field():
            def it_passes_against_the_v3_record_of_the_old_shape(tmp_path: Any):
                # The record predates ``command_id``. The back-fill injects
                # it, so the gate sends no placeholder and still revives.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoBackfill])
                baseline = _baseline_v3_file(
                    tmp_path,
                    events=(
                        (
                            DecoBackfill.__module__,
                            "DecoBackfill.Persist.Persisted",
                            ["note"],
                        ),
                    ),
                )

                assert_all_baselined_revive(serde, baseline)

        def when_the_baseline_lists_a_retired_identity():
            def without_a_migration():
                def it_fails_naming_the_retired_remedy(tmp_path: Any):
                    from langgraph_events.serde.migrations import (
                        assert_all_baselined_revive,
                    )

                    serde = NamespaceAwareSerde(namespaces=[Retiring])
                    baseline = _baseline_v3_file(
                        tmp_path,
                        retired=(("ghost.mod", "Ghost.Gone", ["x"]),),
                    )

                    with pytest.raises(AssertionError) as excinfo:
                        assert_all_baselined_revive(serde, baseline)

                    message = str(excinfo.value)
                    assert "ghost.mod:Ghost.Gone (retired) -> " in message
                    assert "delete its `retired` entry" in message

            def with_a_tombstone_declaring_the_recorded_fields():
                def it_passes(tmp_path: Any):
                    from langgraph_events.serde.migrations import (
                        assert_all_baselined_revive,
                    )

                    serde = NamespaceAwareSerde(namespaces=[Retiring])
                    baseline = _baseline_v3_file(
                        tmp_path,
                        retired=(
                            (
                                Retiring.__module__,
                                "Retiring.ApprovalRequired",
                                ["order_id"],
                            ),
                        ),
                    )

                    assert_all_baselined_revive(serde, baseline)

            def without_a_field_record():
                def it_sends_required_placeholders_only(tmp_path: Any):
                    # A retired entry hand-added for an identity erased
                    # before v3 has no ``fields``. The tombstone's required
                    # ``order_id`` still gets its placeholder.
                    from langgraph_events.serde.migrations import (
                        assert_all_baselined_revive,
                    )

                    serde = NamespaceAwareSerde(namespaces=[Retiring])
                    baseline = _baseline_v3_file(
                        tmp_path,
                        retired=(
                            (Retiring.__module__, "Retiring.ReviewRequired", None),
                        ),
                    )

                    assert_all_baselined_revive(serde, baseline)

        def when_the_live_class_has_required_fields():
            def it_does_not_spuriously_fail(tmp_path: Any):
                # NestedPersist.Persist.Persisted has a required
                # ``command_id`` (no default). The helper must synthesize
                # placeholder kwargs for required fields so a perfectly
                # healthy migration table is not flagged as a failure.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[NestedPersist])
                baseline = _baseline_file(
                    tmp_path,
                    (NestedPersist.__module__, "NestedPersist.Persist.Persisted"),
                )

                assert_all_baselined_revive(serde, baseline)

    def describe_assert_all_baselined_resolve():
        # Resolution-only gate: proves every baselined identity still resolves
        # to a live Event class (rename-aware import walk) without constructing
        # it. Catches renames/removals that ``cover`` (namespace-walk-scoped)
        # misses, with none of the construction-time validation false positives
        # that trip ``revive``.

        def when_baseline_identity_is_still_live():
            def it_passes_silently(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                )

                assert_all_baselined_resolve(serde, baseline)

        def when_a_rename_is_covered_by_migrate_from():
            def it_resolves_through_the_rename(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persisted"),
                )

                assert_all_baselined_resolve(serde, baseline)

        def when_a_baselined_identity_is_only_reachable_in_scope():
            def it_resolves_through_the_scope(tmp_path: Any):
                # The gate answers "would the read path reach a live Event"
                # — so it has to resolve the way the read path does, scope
                # first. Import-only, a function-local namespace fails a
                # gate its own serde revives perfectly well (#150).
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                )

                trading = _local_trading()
                serde = NamespaceAwareSerde(namespaces=[trading])
                baseline = _baseline_file(
                    tmp_path,
                    (
                        trading.Place.Placed.__module__,
                        trading.Place.Placed.__qualname__,
                    ),
                )

                assert_all_baselined_resolve(serde, baseline)

        def when_a_live_event_has_construction_time_validation():
            def it_passes_where_revive_trips(tmp_path: Any):
                # ValidatedReorg.Persist.Persisted rejects an empty/None
                # ``name`` in __post_init__. ``revive`` builds it with a None
                # placeholder and raises; ``resolve`` never constructs, so the
                # identity is proven reachable with no false positive.
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[ValidatedReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (ValidatedReorg.__module__, "ValidatedReorg.Persist.Persisted"),
                )

                with pytest.raises(AssertionError):
                    assert_all_baselined_revive(serde, baseline)

                assert_all_baselined_resolve(serde, baseline)

        def when_the_baseline_lists_an_unmigrated_retired_identity():
            def it_fails_naming_the_retired_remedy(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                )

                serde = NamespaceAwareSerde(namespaces=[Retiring])
                baseline = _baseline_v3_file(
                    tmp_path, retired=(("ghost.mod", "Ghost.Gone", ["x"]),)
                )

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_resolve(serde, baseline)

                message = str(excinfo.value)
                assert "ghost.mod:Ghost.Gone (retired) -> " in message
                assert "delete its `retired` entry" in message

        def when_a_baselined_identity_no_longer_resolves():
            def it_raises_naming_the_missing_identity(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_resolve,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                    ("ghost.mod", "Ghost.Gone"),
                )

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_resolve(serde, baseline)

                assert "Ghost.Gone" in str(excinfo.value)
                assert "DecoReorg.Persist.Persisted" not in str(excinfo.value)
                assert "add @migrate_from" in str(excinfo.value)


def describe_UnreachableMigrationWarning():
    # A @migrate_from-decorated class that namespaces=/events= never
    # reaches contributes nothing — no error, no warning, before this
    # (#164). The fixture lives in its own module
    # (_migrate_from_warning_fixture.py) so its module-level orphan
    # class is never scanned by an unrelated test's serde construction.

    def when_a_decorated_class_is_not_passed_to_the_serde():
        def it_warns_naming_the_class():
            import _migrate_from_warning_fixture as fx

            from langgraph_events.serde import UnreachableMigrationWarning

            with pytest.warns(UnreachableMigrationWarning, match="OrphanTombstone"):
                NamespaceAwareSerde(namespaces=(fx.Gate,))

    def when_the_decorated_class_is_passed_via_events():
        def it_does_not_warn():
            import _migrate_from_warning_fixture as fx

            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                NamespaceAwareSerde(namespaces=(fx.Gate,), events=(fx.OrphanTombstone,))

    def when_no_namespace_reaches_the_decorated_classs_module():
        def it_does_not_warn():
            # The fixture module is never passed at all here — out of
            # scope by design, not a bug; must not be flagged.
            class _Unrelated(Namespace):
                class Placeholder(Interrupted):
                    pass

            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                NamespaceAwareSerde(namespaces=(_Unrelated,))


def describe_public_serde_surface():
    # The common evolution path is decorator-first: @migrate_from plus,
    # rarely, Migration.rename/.add_field sugar. Raw RenameEvent/AddField
    # constructors are an escape hatch for composite multi-op authoring —
    # they stay in serde.migrations but are off the top-level serde
    # re-export so the surface most users see stays small.

    def when_importing_from_the_top_level_serde_package():
        def it_does_not_expose_raw_operation_constructors():
            import langgraph_events.serde as serde_pkg

            assert not hasattr(serde_pkg, "RenameEvent")
            assert not hasattr(serde_pkg, "AddField")
            assert not hasattr(serde_pkg, "TransformFields")
            assert not hasattr(serde_pkg, "SplitEvent")

        def it_still_exposes_the_decorator_and_sugar_tier():
            import langgraph_events.serde as serde_pkg

            for name in (
                "NamespaceAwareSerde",
                "Migration",
                "migrate_from",
                "transform_fields",
                "split_event",
                "synthesize_legacy_payload",
                "assert_all_baselined_revive",
            ):
                assert hasattr(serde_pkg, name), name

    def when_the_composite_escape_hatch_is_needed():
        def it_keeps_raw_ops_importable_from_serde_migrations():
            from langgraph_events.serde.migrations import (
                AddField,
                RenameEvent,
                SplitEvent,
                TransformFields,
            )

            assert RenameEvent is not None
            assert AddField is not None
            assert TransformFields is not None
            assert SplitEvent is not None


def describe_detect_cli():
    # `python -m langgraph_events.serde.migrations <module:factory>
    # <baseline>` — makes "forgot a migration" a non-zero exit (a build
    # failure) outside pytest, not just a prod incident.

    def when_the_graph_matches_the_baseline():
        def it_exits_zero(tmp_path: Any):
            from langgraph_events.serde.migrations.__main__ import main
            from langgraph_events.serde.migrations.detect import write_baseline

            graph = _build_decoreorg_graph()
            baseline = tmp_path / "baseline.json"
            write_baseline(graph, baseline)

            code = main(["test_serde:_build_decoreorg_graph", str(baseline)])

            assert code == 0

    def when_the_baseline_diverges_from_the_graph():
        def it_exits_nonzero(tmp_path: Any):
            from langgraph_events.serde.migrations.__main__ import main

            baseline = _baseline_file(tmp_path, ("ghost.mod", "Ghost.Gone"))

            code = main(["test_serde:_build_decoreorg_graph", str(baseline)])

            assert code == 1

    def when_arguments_are_missing():
        def it_returns_a_usage_code(tmp_path: Any):
            from langgraph_events.serde.migrations.__main__ import main

            assert main(["only-one-arg"]) == 2

    def when_invoked_as_a_module():
        # The other tests call main() directly and so cannot catch a wrong
        # `python -m ...` target. `__main__.py` lives in the `migrations`
        # package, so the working entrypoint is
        # `python -m langgraph_events.serde.migrations`; the program's own
        # usage message must advertise that, not the inert
        # `...migrations.detect` (a module with no __main__ guard).
        def it_advertises_the_working_entrypoint():
            import subprocess
            import sys

            result = subprocess.run(  # noqa: S603 — fixed argv, no shell
                [sys.executable, "-m", "langgraph_events.serde.migrations"],
                capture_output=True,
                check=False,  # asserting the non-zero usage exit ourselves
            )

            assert result.returncode == 2
            assert b"usage:" in result.stderr
            assert b"migrations.detect" not in result.stderr


def describe_replay_reducer():
    # `replay_reducer` names the recovery pattern when a reducer's
    # projection function or output shape changes between releases.
    # The cached channel value is stale, but events are the source of
    # truth and can be replayed through the reducer's `seed()`.

    def when_called_on_a_list_reducer():
        def it_rebuilds_the_channel_value_via_seed():
            from langgraph_events import Reducer
            from langgraph_events.serde import replay_reducer

            reducer = Reducer(
                name="notes",
                event_type=Persona.Approve.Approved,
                fn=lambda e: [e.note],
            )
            events = [
                Persona.Approve.Approved(note="first"),
                Persona.Approve.Approved(note="second"),
                # Mismatched event type — should be filtered out by
                # the reducer's event_type predicate.
                Story.Approve.Approved(note="ignored"),
            ]

            rebuilt = replay_reducer(reducer, events)

            assert rebuilt == ["first", "second"]
            # Matches what `seed` returns directly — confirms the helper
            # is a transparent wrapper, not adding accidental behaviour.
            assert rebuilt == reducer.seed(events)


def describe_from_namespaces_serde_auto_wiring():
    def when_the_checkpointer_carries_the_default_serde():
        def it_wraps_it_in_a_namespace_scoped_serde():
            # The user should only need @migrate_from on the class — not
            # know NamespaceAwareSerde exists or thread a namespace tuple
            # into it. from_namespaces already has both the domains and the
            # checkpointer, so it wires the serde itself.
            graph = EventGraph.from_namespaces(DecoReorg, checkpointer=MemorySaver())

            serde = graph._checkpointer.serde
            assert isinstance(serde, NamespaceAwareSerde)

            payload = synthesize_legacy_payload(
                DecoReorg.__module__, "DecoReorg.Persisted", {"note": "x"}
            )
            revived = serde.loads_typed(payload)

            assert isinstance(revived, DecoReorg.Persist.Persisted)
            assert revived.note == "x"

    def when_the_user_supplies_their_own_namespace_aware_serde():
        def it_is_left_untouched():
            # Opt-out: a user-constructed serde (possibly carrying
            # hand-authored migrations=) must win over auto-wiring.
            own = NamespaceAwareSerde(namespaces=[DecoReorg])

            graph = EventGraph.from_namespaces(
                DecoReorg, checkpointer=MemorySaver(serde=own)
            )

            assert graph._checkpointer.serde is own


def describe_backfill():
    # @backfill is the class-scoped, auto-collected sibling of
    # @migrate_from for the "added a now-required field" case: the
    # back-fill value lives on the class, not in a remote migrations=
    # list, and is picked up by the same namespace walk.

    def when_a_legacy_payload_predates_the_field():
        def it_injects_the_decorated_default_like_hand_authored_AddField():
            from langgraph_events.serde.migrations import Migration

            via_decorator = NamespaceAwareSerde(namespaces=[DecoBackfill])
            via_hand_authored = NamespaceAwareSerde(
                migrations=[
                    Migration.add_field(
                        target=DecoBackfill.Persist.Persisted,
                        field="command_id",
                        default="legacy",
                    )
                ]
            )
            legacy = synthesize_legacy_payload(
                DecoBackfill.__module__,
                "DecoBackfill.Persist.Persisted",
                {"note": "n"},  # written before command_id existed
            )

            d = via_decorator.loads_typed(legacy)
            h = via_hand_authored.loads_typed(legacy)

            assert d == h
            assert isinstance(d, DecoBackfill.Persist.Persisted)
            assert d.command_id == "legacy"
            assert d.note == "n"

    def when_the_graph_is_built_via_from_namespaces():
        def it_back_fills_via_auto_wiring():
            # The headline: @backfill rejoins the no-ceremony path opened
            # by Finding 1 — only the decorator on the class plus a
            # checkpointer; no NamespaceAwareSerde, no migrations= list.
            graph = EventGraph.from_namespaces(DecoBackfill, checkpointer=MemorySaver())

            revived = graph._checkpointer.serde.loads_typed(
                synthesize_legacy_payload(
                    DecoBackfill.__module__,
                    "DecoBackfill.Persist.Persisted",
                    {"note": "n"},
                )
            )

            assert isinstance(revived, DecoBackfill.Persist.Persisted)
            assert revived.command_id == "legacy"

    def when_the_class_also_carries_migrate_from():
        def it_renames_then_back_fills_in_one_pass():
            serde = NamespaceAwareSerde(namespaces=[DecoBackfillRenamed])

            # Written under the OLD qualname, before command_id existed.
            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoBackfillRenamed.__module__,
                    "DecoBackfillRenamed.Persisted",
                    {"note": "old"},
                )
            )

            assert isinstance(revived, DecoBackfillRenamed.Persist.Persisted)
            assert revived.command_id == "legacy"
            assert revived.note == "old"

    def when_the_decorated_default_is_mutable():
        def it_is_rejected_by_the_shared_AddField_guard():
            with pytest.raises(ValueError, match="mutable"):
                NamespaceAwareSerde(namespaces=[DecoBackfillBadDefault])

    def when_a_subclass_inherits_a_backfilled_parent():
        def it_does_not_emit_a_back_fill_for_the_subclass():
            # Mirrors the @migrate_from SubLeak guard: the marker must not
            # leak through MRO. Only Parent's identity may key the
            # addfield table.
            serde = NamespaceAwareSerde(namespaces=[BackfillSubLeak])

            targets = [qn for (_, qn) in serde._addfield_table]
            assert targets == ["BackfillSubLeak.Parent"]

    def when_backfill_decorators_are_stacked():
        def it_injects_every_declared_field():
            serde = NamespaceAwareSerde(namespaces=[DecoBackfillStack])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoBackfillStack.__module__,
                    "DecoBackfillStack.Persist.Persisted",
                    {"note": "n"},  # both command_id and flag absent
                )
            )

            assert revived.command_id == "legacy"
            assert revived.flag is False
            assert revived.note == "n"


def describe_origin_scoped_backfill():
    # ``backfill=`` on @migrate_from scopes a field default to payloads
    # that originated under THAT decorator's historic qualname — the
    # fan-in case (#101) where N classes collapse into one and a
    # class-global @backfill could not tell the origins apart.

    def when_collapsed_origins_pin_their_own_discriminator():
        def it_injects_a_distinct_default_per_origin():
            serde = NamespaceAwareSerde(namespaces=[DecoFanIn])

            for origin, expected in [
                ("PersonaOld.Approve.Approved", "persona"),
                ("StoryOld.Approve.Approved", "story"),
                ("ScenarioOld.Approve.Approved", "scenario"),
            ]:
                revived = serde.loads_typed(
                    synthesize_legacy_payload(
                        DecoFanIn.__module__, origin, {"entity_id": "e-1"}
                    )
                )

                assert isinstance(revived, DecoFanIn.Approved)
                assert revived.entity_type == expected
                assert revived.entity_id == "e-1"

    def when_the_payload_carries_an_explicit_value():
        def it_is_preserved_over_the_origin_fill():
            serde = NamespaceAwareSerde(namespaces=[DecoFanIn])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoFanIn.__module__,
                    "PersonaOld.Approve.Approved",
                    {"entity_id": "e-1", "entity_type": "story"},
                )
            )

            assert revived.entity_type == "story"

    def when_an_origin_has_no_scoped_fill():
        def it_falls_back_to_the_class_global_backfill():
            serde = NamespaceAwareSerde(namespaces=[DecoFanInFallback])

            scoped = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoFanInFallback.__module__, "PersonaOldFB.Approved", {}
                )
            )
            unscoped = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoFanInFallback.__module__, "LegacyOldFB.Approved", {}
                )
            )

            assert scoped.entity_type == "persona"
            assert unscoped.entity_type == "unknown"

    def when_a_subclass_inherits_an_origin_backfilled_parent():
        def it_does_not_emit_a_fill_for_the_subclass():
            # Mirrors the @migrate_from / @backfill SubLeak guards: the
            # marker must not leak through MRO. A leak would re-emit
            # Parent's origin AddField for Child and trip the
            # duplicate-fill guard at construction.
            serde = NamespaceAwareSerde(namespaces=[OriginBackfillSubLeak])

            assert list(serde._origin_addfield_table) == [
                (OriginBackfillSubLeak.__module__, "OriginBackfillSubLeak.OldParent")
            ]

    def when_the_scoped_value_is_mutable():
        def it_is_rejected_at_decoration_steering_to_the_raw_escape_hatch():
            # The error must fire at decoration (the decorator holds the
            # dict in its hand) and steer to the escape hatch that exists
            # for the dict form — Migration.add_field keyed on the historic
            # identity — not to AddField's `default_factory=` kwarg, which
            # backfill={...} has no way to spell.
            with pytest.raises(ValueError, match=r"Migration\.add_field"):

                @migrate_from("BadValueOld.Approved", backfill={"tags": []})
                class Bad(DomainEvent):
                    tags: tuple[str, ...] = ()

    def when_stacked_decorators_repeat_an_origin():
        def it_is_rejected_at_decoration_time():
            # Without this guard the duplicate produces a chain
            # Old → Old → current whose "Duplicate rename source" error
            # names the same auto-migration twice with one arrow pointing
            # the origin at itself — chain vocabulary for what is an
            # origin duplication.
            with pytest.raises(ValueError, match=r"[Dd]uplicate origin"):

                class DupHolder(Namespace):
                    @migrate_from("DupOld.Approved")
                    @migrate_from("DupOld.Approved")
                    class Dup(DomainEvent):
                        note: str = ""

        def it_is_rejected_in_the_multi_arg_form_too():
            with pytest.raises(ValueError, match=r"[Dd]uplicate origin"):

                class DupArgsHolder(Namespace):
                    @migrate_from("DupOld.A", "DupOld.A")
                    class DupArgs(DomainEvent):
                        note: str = ""

    def when_a_fill_names_an_unknown_field():
        def it_is_rejected_at_serde_construction():
            # A typo'd field name would otherwise surface as a dataclass
            # TypeError at first production read — the worst possible time.
            with pytest.raises(ValueError, match="entity_typ"):
                NamespaceAwareSerde(namespaces=[DecoFanInTypoField])

        def it_guards_the_class_global_decorator_too():
            with pytest.raises(ValueError, match="command_idd"):
                NamespaceAwareSerde(namespaces=[DecoBackfillTypoField])

    def when_a_fill_names_an_init_false_field():
        def it_is_rejected_at_serde_construction():
            # The encoder never writes the field and the constructor rejects
            # it as a kwarg, so the fill can only break every read.
            with pytest.raises(ValueError, match="has no field 'derived'"):
                NamespaceAwareSerde(namespaces=[DecoBackfillInitFalse])

    def when_legacy_write_meets_origin_scoped_fills():
        def it_is_rejected_at_serde_construction():
            # legacy_write relabels EVERY instance under the oldest
            # historic identity — persona, story, and scenario alike —
            # exactly the failure the docs admonition forbids. Enforce it
            # at construction instead of documenting around it.
            with pytest.raises(ValueError, match="legacy_write"):
                NamespaceAwareSerde(namespaces=[DecoFanIn], legacy_write=True)

    def when_backfill_is_an_empty_dict():
        def it_is_rejected_at_decoration_time():
            # backfill={} previously passed the `is not None` chain guard
            # and was then silently dropped — the two guards disagreed
            # about whether an empty dict counts.
            with pytest.raises(ValueError, match="at least one field"):

                @migrate_from("EmptyOld.Approved", backfill={})
                class Empty(DomainEvent):
                    note: str = ""

    def when_a_raw_AddField_targets_a_historic_identity():

        def _new_era_serde() -> NamespaceAwareSerde:
            """Serde whose only fill is keyed on the historic ``ChainScoped.New``."""
            from langgraph_events.serde.migrations import Migration

            return NamespaceAwareSerde(
                namespaces=[ChainScoped],
                migrations=[
                    Migration.add_field(
                        name="new-era-fill",
                        module=ChainScoped.__module__,
                        qualname="ChainScoped.New",
                        field="source",
                        default="new-era",
                    )
                ],
            )

        def it_applies_before_the_rename():
            revived = _new_era_serde().loads_typed(
                synthesize_legacy_payload(ChainScoped.__module__, "ChainScoped.New", {})
            )

            assert isinstance(revived, ChainScoped.Persisted)
            assert revived.source == "new-era"

        def it_does_not_cascade_to_earlier_chain_members():
            # Exact-origin contract: on the chain Old → New → current, a
            # fill keyed on New must not apply to payloads written under
            # Old. "Every era" is class-global @backfill's job.
            revived = _new_era_serde().loads_typed(
                synthesize_legacy_payload(ChainScoped.__module__, "ChainScoped.Old", {})
            )

            assert revived.source == ""

        def it_supports_a_per_origin_default_factory():
            # The escape hatch backfill= (immutable values only) points to:
            # a hand-authored AddField keyed on the historic identity with
            # a default_factory, fresh instance per read.
            from langgraph_events.serde.migrations import Migration

            serde = NamespaceAwareSerde(
                namespaces=[FactoryScoped],
                migrations=[
                    Migration.add_field(
                        name="origin-tags",
                        module=FactoryScoped.__module__,
                        qualname="FactoryScoped.Old",
                        field="tags",
                        default_factory=lambda: ["legacy"],
                    )
                ],
            )
            payload = synthesize_legacy_payload(
                FactoryScoped.__module__, "FactoryScoped.Old", {"note": "n"}
            )

            first = serde.loads_typed(payload)
            second = serde.loads_typed(payload)

            assert first.tags == ["legacy"]
            assert first.tags is not second.tags

    def when_a_baseline_contains_collapsed_origins():
        def it_passes_the_resolve_and_revive_gates(tmp_path: Any):
            # The acceptance sketch from #101: a prior release's baseline
            # still lists the three per-entity identities; both gates must
            # pass with zero per-event test code.
            from langgraph_events.serde.migrations import (
                assert_all_baselined_resolve,
                assert_all_baselined_revive,
            )

            serde = NamespaceAwareSerde(namespaces=[DecoFanIn])
            baseline = _baseline_file(
                tmp_path,
                (DecoFanIn.__module__, "PersonaOld.Approve.Approved"),
                (DecoFanIn.__module__, "StoryOld.Approve.Approved"),
                (DecoFanIn.__module__, "ScenarioOld.Approve.Approved"),
            )

            assert_all_baselined_resolve(serde, baseline)
            assert_all_baselined_revive(serde, baseline)

        def it_exercises_the_origin_fill_instead_of_a_placeholder(tmp_path: Any):
            # ValidatedFanIn's __post_init__ rejects a None discriminator —
            # the gate must let the origin-scoped fill supply the value
            # rather than masking it with a required-field placeholder
            # (which would both hide a broken fill and spuriously fail
            # validated classes whose backfill is perfectly healthy).
            from langgraph_events.serde.migrations import (
                assert_all_baselined_revive,
            )

            serde = NamespaceAwareSerde(namespaces=[ValidatedFanIn])
            baseline = _baseline_file(
                tmp_path,
                (ValidatedFanIn.__module__, "ValidatedFanInOld.Approved"),
            )

            assert_all_baselined_revive(serde, baseline)

        def it_exercises_class_global_fills_the_same_way(tmp_path: Any):
            from langgraph_events.serde.migrations import (
                assert_all_baselined_revive,
            )

            serde = NamespaceAwareSerde(namespaces=[ValidatedGlobalBackfill])
            baseline = _baseline_file(
                tmp_path,
                (
                    ValidatedGlobalBackfill.__module__,
                    "ValidatedGlobalBackfill.Approved",
                ),
            )

            assert_all_baselined_revive(serde, baseline)

    def when_the_fill_value_is_broken():
        def it_fails_the_revive_gate_loudly(tmp_path: Any):
            from langgraph_events.serde.migrations import (
                assert_all_baselined_revive,
            )

            serde = NamespaceAwareSerde(namespaces=[ValidatedBadFill])
            baseline = _baseline_file(
                tmp_path,
                (ValidatedBadFill.__module__, "ValidatedBadFill.Approved"),
            )

            # The sweep names the offending identity; the inner
            # ``__post_init__`` text is swallowed by ormsgpack's generic
            # "ext_hook failed" (pre-existing diagnostic limitation).
            with pytest.raises(AssertionError, match=r"ValidatedBadFill\.Approved"):
                assert_all_baselined_revive(serde, baseline)

    def when_two_fills_target_the_same_origin_field():
        def it_is_rejected_at_serde_construction():
            # Without a guard the first op silently wins. Reachable via a
            # decorator fill colliding with a hand-authored one (stacked
            # duplicate decorators are caught earlier, at decoration).
            # Mirrors the duplicate-rename-source diagnostic. Note an
            # edited-qualname/un-edited-VALUE slip is undetectable by
            # design — values carry no intent to compare against.
            from langgraph_events.serde.migrations import Migration

            conflicting = Migration.add_field(
                name="conflicting-fill",
                module=DecoFanIn.__module__,
                qualname="PersonaOld.Approve.Approved",
                field="entity_type",
                default="story",
            )

            with pytest.raises(ValueError, match="Duplicate AddField"):
                NamespaceAwareSerde(namespaces=[DecoFanIn], migrations=[conflicting])

    def when_backfill_rides_a_multi_qualname_chain_decorator():
        def it_is_rejected_at_decoration_time():
            # ``@migrate_from("A", "B")`` declares a temporal chain — a
            # back-fill attached to it is ambiguous about which origin it
            # belongs to. Scoped fills are one decorator per origin.
            with pytest.raises(ValueError, match="exactly one historic"):

                @migrate_from("Chain.A", "Chain.B", backfill={"x": 1})
                class Ambiguous(DomainEvent):
                    x: int


# Module-level fixtures for the renamed-inline-command resume test (#103).
# ``OldLocker`` plays the previous release's class; the test deletes it
# mid-flight (a rename means the old definition is gone) and restores it
# afterwards.
class _RelockPause(Interrupted):
    pass


class _RelockGo(IntegrationEvent):
    pass


class OldLocker(Namespace):
    class Persist(Command):
        def handle(self) -> _RelockPause:
            return _RelockPause()


class Locker(Namespace):
    @migrate_from("OldLocker.Persist")
    class Persist(Command):
        previously: ClassVar = ("OldLocker.Persist",)

        def handle(self) -> _RelockPause:
            return _RelockPause()


def describe_renamed_inline_command_resume():
    # Renaming a command class is BOTH renames at once: the node identity in
    # snapshot.next and the event class of the checkpointed payload. The
    # alias node dispatches by isinstance against the new class, so resume
    # only recovers when previously= (node) and @migrate_from (payload, via
    # a namespace-aware serde) are declared together — the docs warning in
    # event-migrations.md, enforced here.

    def when_the_command_class_itself_is_renamed():
        def with_both_rename_tracks_declared():
            def it_resumes_a_checkpoint_paused_under_the_old_class():
                import sys

                @on(_RelockGo)
                def go(event: _RelockGo) -> None:
                    return None

                saver = MemorySaver(serde=NamespaceAwareSerde())
                before = EventGraph([OldLocker.Persist, go], checkpointer=saver)
                cfg = {"configurable": {"thread_id": "relock-1"}}
                before.invoke(OldLocker.Persist(), config=cfg)
                assert before.get_state(cfg).is_interrupted

                mod = sys.modules[OldLocker.__module__]
                old_ns = OldLocker
                try:
                    # The new release: the old class definition is gone, the
                    # surviving class declares both rename tracks.
                    delattr(mod, "OldLocker")
                    saver.serde = NamespaceAwareSerde(namespaces=[Locker])
                    after = EventGraph([Locker.Persist, go], checkpointer=saver)

                    log = after.resume(_RelockGo(), config=cfg)

                    assert log.has(Resumed)
                    # The checkpointed payload revived as the renamed class.
                    assert log.has(Locker.Persist)
                finally:
                    mod.OldLocker = old_ns


def describe_assert_all_baselined_handlers_cover():
    # The handler analog of the event coverage gates: every handler node name
    # the previous release wrote must still be live or covered by an
    # @on(previously=...) alias, else an interrupted checkpoint paused at the
    # old node would silently drop on resume.
    # Aliased under ``_`` so pytest-describe does not collect the imported
    # callables as tests of this block.
    from langgraph_events.serde.migrations import (
        assert_all_baselined_handlers_cover as _assert_handlers_cover,
    )
    from langgraph_events.serde.migrations.detect import (
        CoverageError,
        HandlerCoverageError,
    )
    from langgraph_events.serde.migrations.detect import (
        write_baseline as _write_baseline,
    )

    def _baseline_for(graph: EventGraph, tmp_path: Any) -> Any:
        """Write ``graph``'s handler baseline under *tmp_path* and return it."""
        baseline = tmp_path / "b.json"
        _write_baseline(graph, baseline)
        return baseline

    def when_every_baselined_handler_is_live():
        def it_passes_silently(tmp_path: Any):
            @on(Started)
            def place(event: Started) -> None:
                return None

            graph = EventGraph([place])
            baseline = _baseline_for(graph, tmp_path)

            _assert_handlers_cover(graph, baseline)

    def when_a_handler_is_renamed():
        def with_previously_declared():
            def it_passes(tmp_path: Any):
                @on(Started)
                def place(event: Started) -> None:
                    return None

                baseline = _baseline_for(EventGraph([place]), tmp_path)

                @on(Started, previously="place")
                def submit(event: Started) -> None:
                    return None

                _assert_handlers_cover(EventGraph([submit]), baseline)

        def without_previously_declared():
            def it_raises_naming_the_lost_handler(tmp_path: Any):
                @on(Started)
                def place(event: Started) -> None:
                    return None

                baseline = _baseline_for(EventGraph([place]), tmp_path)

                @on(Started)
                def submit(event: Started) -> None:
                    return None

                with pytest.raises(HandlerCoverageError, match="place") as excinfo:
                    _assert_handlers_cover(EventGraph([submit]), baseline)
                # sibling of MigrationCoverageError under the shared base
                assert isinstance(excinfo.value, CoverageError)
                assert excinfo.value.uncovered == ("place",)
                # Grammar must agree with the count: one handler, not "resolve".
                assert "1 baselined handler no longer resolves to" in str(excinfo.value)

    def when_a_reactor_is_replaced_by_an_inline_command():
        # The command's ``previously`` class attribute feeds the same
        # HandlerMeta.previous_names the gate unions into its reachable set.
        def with_previously_declared():
            def it_passes(tmp_path: Any):
                class _Replat(Namespace):
                    class Persist(Command):
                        previously: ClassVar = ("persist",)

                        def handle(self) -> None:
                            return None

                @on(Started, node_name="persist")
                def persist_reactor(event: Started) -> None:
                    return None

                baseline = _baseline_for(EventGraph([persist_reactor]), tmp_path)

                _assert_handlers_cover(EventGraph([_Replat.Persist]), baseline)

        def without_previously_declared():
            def it_raises_naming_the_lost_handler(tmp_path: Any):
                class _Replat2(Namespace):
                    class Persist(Command):
                        def handle(self) -> None:
                            return None

                @on(Started, node_name="persist")
                def persist_reactor(event: Started) -> None:
                    return None

                baseline = _baseline_for(EventGraph([persist_reactor]), tmp_path)

                with pytest.raises(HandlerCoverageError, match="persist"):
                    _assert_handlers_cover(EventGraph([_Replat2.Persist]), baseline)

    def when_baseline_recorded_old_positional_inline_command_names():
        # Pre-#97 baselines recorded inline command handlers by their
        # positional ``handle``/``handle_2`` names. After the fix those nodes
        # are keyed by the command qualname, so the old names no longer
        # resolve — the gate must fire, forcing a one-time baseline rebuild.

        def it_raises_until_the_baseline_is_regenerated(tmp_path: Any):
            import json

            graph = EventGraph([Persona.Approve, Story.Approve])
            baseline = tmp_path / "b.json"
            baseline.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "events": [],
                        "handlers": [{"name": "handle"}, {"name": "handle_2"}],
                    }
                )
            )

            with pytest.raises(HandlerCoverageError) as excinfo:
                _assert_handlers_cover(graph, baseline)
            assert excinfo.value.uncovered == ("handle", "handle_2")

            # The documented one-time migration: regenerate the baseline.
            _write_baseline(graph, baseline)
            _assert_handlers_cover(graph, baseline)

    def when_baseline_predates_handler_tracking():
        def it_loads_a_v1_baseline_and_passes(tmp_path: Any):
            import json

            @on(Started)
            def place(event: Started) -> None:
                return None

            baseline = tmp_path / "v1.json"
            baseline.write_text(json.dumps({"version": 1, "events": []}))

            _assert_handlers_cover(EventGraph([place]), baseline)


# Fixtures for ``TransformFields``. Each class dropped, merged or retyped a
# field after payloads were written, so a stored payload no longer matches
# the live constructor. The transforms are module-level so a decorator
# fixture below can reuse them.
def _dropping(key: str) -> Any:
    """A transform that removes *key* when present."""

    def _transform(kw: dict[str, Any]) -> dict[str, Any]:
        kw.pop(key, None)
        return kw

    return _transform


_drop_legacy_flag = _dropping("legacy_flag")


def _recording(calls: list[str], label: str) -> Any:
    """A transform that appends *label* to *calls* and changes nothing."""

    def _transform(kw: dict[str, Any]) -> dict[str, Any]:
        calls.append(label)
        return kw

    return _transform


def _merge_name(kw: dict[str, Any]) -> dict[str, Any]:
    first = kw.pop("first", None)
    last = kw.pop("last", None)
    if first is not None or last is not None:
        kw["name"] = f"{first or ''} {last or ''}".strip()
    return kw


def _int_count(kw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(kw.get("count"), str):
        kw["count"] = int(kw["count"])
    return kw


class Reshaped(Namespace):
    class Trimmed(DomainEvent):
        """Dropped ``legacy_flag``."""

        note: str = ""

    class Named(DomainEvent):
        """Merged ``first`` and ``last`` into ``name``."""

        name: str = ""

    @transform_fields(_int_count)
    class Counted(DomainEvent):
        """``count`` was a ``str``, now an ``int``."""

        count: int = 0


def _upper_draft(kw: dict[str, Any]) -> dict[str, Any]:
    """Idempotent, and visible on a payload the current release wrote."""
    if "draft" in kw:
        kw["draft"] = kw["draft"].upper()
    return kw


# A class-stage transform composed with a rename. The transform is keyed
# on the live identity, so it runs on every era: the historic origin and
# the payloads the current release writes.
class Gate(Namespace):
    @transform_fields(_upper_draft)
    @migrate_from("Gate.OldReviewed")
    class Reviewed(Interrupted):
        draft: str = ""


# An origin-scoped transform: it runs before the rename, on payloads
# written under ``DecoOriginTransform.Old`` only.
class DecoOriginTransform(Namespace):
    @migrate_from("DecoOriginTransform.Old", transform=_drop_legacy_flag)
    class Persisted(DomainEvent):
        note: str = ""


# ``transform=`` and ``backfill=`` on one decorator. The transform runs
# first, then the origin fill supplies the required ``source``.
class DecoOriginBoth(Namespace):
    @migrate_from(
        "DecoOriginBoth.Old", transform=_drop_legacy_flag, backfill={"source": "old"}
    )
    class Persisted(DomainEvent):
        source: str


# A chain A -> B -> Persisted for hand-authored transforms keyed on A (the
# origin stage) and on Persisted (the class stage).
class ChainTransform(Namespace):
    @migrate_from("ChainTransform.A", "ChainTransform.B")
    class Persisted(DomainEvent):
        note: str = ""


def describe_TransformFields():
    # A kwargs-to-kwargs operation. A rename covers a moved class and an
    # AddField covers a gained field. A dropped, merged or retyped field
    # has no operation without this one (#159).

    def when_a_stored_payload_carries_a_dropped_field():
        def it_revives_after_the_transform_drops_it():
            from langgraph_events.serde.migrations import (
                Migration,
                TransformFields,
            )

            serde = NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration(
                        name="drop-legacy-flag",
                        operations=(
                            TransformFields(
                                module=Reshaped.__module__,
                                qualname="Reshaped.Trimmed",
                                transform=_drop_legacy_flag,
                            ),
                        ),
                    )
                ],
            )

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    Reshaped.__module__,
                    "Reshaped.Trimmed",
                    {"note": "n", "legacy_flag": True},
                )
            )

            assert isinstance(revived, Reshaped.Trimmed)
            assert revived.note == "n"

    def when_two_stored_fields_merge_into_one():
        def it_revives_holding_the_merged_value():
            # ``Migration.transform_fields`` is the single-op sugar, beside
            # ``Migration.rename`` and ``Migration.add_field``.
            from langgraph_events.serde.migrations import Migration

            serde = NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration.transform_fields(
                        target=Reshaped.Named, transform=_merge_name
                    )
                ],
            )

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    Reshaped.__module__,
                    "Reshaped.Named",
                    {"first": "Ada", "last": "Lovelace"},
                )
            )

            assert revived == Reshaped.Named(name="Ada Lovelace")

    def when_a_stored_field_changed_type():
        def it_revives_holding_the_retyped_value():
            # ``@transform_fields`` is the class-stage decorator form,
            # collected from ``namespaces=`` like ``@backfill``.
            serde = NamespaceAwareSerde(namespaces=[Reshaped])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    Reshaped.__module__, "Reshaped.Counted", {"count": "3"}
                )
            )

            assert revived == Reshaped.Counted(count=3)

    def when_the_transform_is_keyed_on_the_live_identity():
        # The class stage. It runs after the rename, on every era,
        # so the transform must be idempotent.

        def it_runs_on_a_payload_the_current_release_wrote():
            serde = NamespaceAwareSerde(namespaces=[Gate])

            revived = serde.loads_typed(serde.dumps_typed(Gate.Reviewed(draft="d")))

            assert revived == Gate.Reviewed(draft="D")

        def it_runs_on_a_payload_renamed_onto_the_class():
            serde = NamespaceAwareSerde(namespaces=[Gate])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    Gate.__module__, "Gate.OldReviewed", {"draft": "d"}
                )
            )

            assert revived == Gate.Reviewed(draft="D")

        def it_reaches_an_event_nested_inside_a_Resumed():
            # ``Resumed.interrupted`` is one ext record inside another, so
            # the transform runs through the hook's recursion.
            serde = NamespaceAwareSerde(namespaces=[Gate])
            stored = Resumed(interrupted=Gate.Reviewed(draft="d"))

            revived = serde.loads_typed(serde.dumps_typed(stored))

            assert revived.interrupted == Gate.Reviewed(draft="D")

        def it_runs_once_per_read():
            # A live identity with no rename has the same key before and
            # after the rename step. The class stage must still run once.
            from langgraph_events.serde.migrations import Migration

            calls: list[str] = []
            serde = NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration.transform_fields(
                        target=Reshaped.Trimmed, transform=_recording(calls, "T")
                    )
                ],
            )

            serde.loads_typed(serde.dumps_typed(Reshaped.Trimmed(note="n")))

            assert calls == ["T"]

    def when_the_transform_is_keyed_on_a_rename_source():
        # The origin stage, declared with ``migrate_from(transform=...)``.

        def it_runs_for_that_origin_only():
            serde = NamespaceAwareSerde(namespaces=[DecoOriginTransform])
            module = DecoOriginTransform.__module__

            old_era = serde.loads_typed(
                synthesize_legacy_payload(
                    module, "DecoOriginTransform.Old", {"legacy_flag": True}
                )
            )

            assert old_era == DecoOriginTransform.Persisted()
            with pytest.raises(ValueError, match="legacy_flag"):
                serde.loads_typed(
                    synthesize_legacy_payload(
                        module,
                        "DecoOriginTransform.Persisted",
                        {"legacy_flag": True},
                    )
                )

        def with_two_historic_qualnames():
            def it_is_rejected_at_decoration_time():
                # Same rule as ``backfill=``: a chain is ambiguous about
                # which origin the transform belongs to.
                with pytest.raises(ValueError, match="exactly one historic"):

                    @migrate_from("Chain.A", "Chain.B", transform=_drop_legacy_flag)
                    class Ambiguous(DomainEvent):
                        note: str = ""

    def when_transforms_sit_on_every_step_of_a_chain():
        # Chain A -> B -> Persisted, with a transform keyed on each step.
        # A and B are origin stages, Persisted is the class stage. The run
        # list per era pins which stages run, and in which order.

        def _serde(calls: list[str]) -> NamespaceAwareSerde:
            from langgraph_events.serde.migrations import (
                Migration,
                TransformFields,
            )

            module = ChainTransform.__module__
            return NamespaceAwareSerde(
                namespaces=[ChainTransform],
                migrations=[
                    Migration(
                        name="every-step",
                        operations=tuple(
                            TransformFields(module, qualname, _recording(calls, label))
                            for qualname, label in (
                                ("ChainTransform.A", "T(A)"),
                                ("ChainTransform.B", "T(B)"),
                                ("ChainTransform.Persisted", "T(C)"),
                            )
                        ),
                    )
                ],
            )

        def _runs_for(qualname: str) -> list[str]:
            calls: list[str] = []
            _serde(calls).loads_typed(
                synthesize_legacy_payload(ChainTransform.__module__, qualname, {})
            )
            return calls

        def it_runs_the_A_origin_then_the_class_stage_on_an_A_era_payload():
            assert _runs_for("ChainTransform.A") == ["T(A)", "T(C)"]

        def it_runs_the_B_origin_then_the_class_stage_on_a_B_era_payload():
            assert _runs_for("ChainTransform.B") == ["T(B)", "T(C)"]

        def it_runs_the_class_stage_once_on_a_live_payload():
            calls: list[str] = []
            serde = _serde(calls)

            serde.loads_typed(serde.dumps_typed(ChainTransform.Persisted()))

            assert calls == ["T(C)"]

    def when_the_transform_raises():
        # ormsgpack swallows an exception raised inside the ext-hook and
        # re-raises a bare ``ext_hook failed``. The transform runs inside
        # the hook, so its failure must reach the same ``errors`` channel
        # the resolve failure uses.

        def _serde() -> NamespaceAwareSerde:
            from langgraph_events.serde.migrations import Migration

            def broken(kw: dict[str, Any]) -> dict[str, Any]:
                return {"note": kw["legacy_flag"]}

            return NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration.transform_fields(
                        target=Reshaped.Trimmed, transform=broken
                    )
                ],
            )

        _payload = synthesize_legacy_payload(
            Reshaped.__module__, "Reshaped.Trimmed", {"note": "n"}
        )

        def it_raises_naming_the_stored_identity_and_the_cause():
            with pytest.raises(ValueError) as excinfo:
                _serde().loads_typed(_payload)

            message = str(excinfo.value)
            assert message.startswith(
                f"Cannot revive {Reshaped.__module__}.Reshaped.Trimmed: "
                f"TransformFields raised KeyError: 'legacy_flag'."
            )
            # The gate sends None for a dropped field, so the remedy must
            # cover a key that is present but None, not only an absent one.
            assert "kw.pop('x', None) and a None check" in message

        def with_tolerate_unresolved():
            def it_degrades_and_collects_the_identity():
                from langgraph_events.serde._jsonplus import UnrevivedIdentity

                serde = _serde()
                with serde.tolerate_unresolved() as unresolved:
                    revived = serde.loads_typed(_payload)

                placeholder = UnrevivedIdentity(Reshaped.__module__, "Reshaped.Trimmed")
                assert revived == placeholder
                assert unresolved == [placeholder]

    def when_the_transform_returns_a_non_dict():
        def it_raises_naming_the_stored_identity_and_the_type():
            from langgraph_events.serde.migrations import Migration

            serde = NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration.transform_fields(target=Reshaped.Trimmed, transform=list)
                ],
            )

            with pytest.raises(ValueError) as excinfo:
                serde.loads_typed(
                    synthesize_legacy_payload(
                        Reshaped.__module__, "Reshaped.Trimmed", {"note": "n"}
                    )
                )

            assert str(excinfo.value).startswith(
                f"Cannot revive {Reshaped.__module__}.Reshaped.Trimmed: "
                f"TransformFields raised TypeError: transform returned list"
            )

    def when_two_transforms_target_one_identity():
        def it_is_rejected_at_serde_construction():
            # The read path runs one transform per stage. A second one
            # would be silently ignored. Compose the steps in one callable.
            from langgraph_events.serde.migrations import Migration

            conflicting = Migration.transform_fields(
                name="second", target=Reshaped.Counted, transform=_drop_legacy_flag
            )

            with pytest.raises(ValueError, match="Duplicate TransformFields"):
                NamespaceAwareSerde(namespaces=[Reshaped], migrations=[conflicting])

        def with_two_unnamed_migrations():
            def it_asks_for_names_instead_of_naming_nothing_twice():
                from langgraph_events.serde.migrations import Migration

                first = Migration.transform_fields(
                    target=Reshaped.Trimmed, transform=_drop_legacy_flag
                )
                second = Migration.transform_fields(
                    target=Reshaped.Trimmed, transform=_int_count
                )

                with pytest.raises(
                    ValueError,
                    match=r"by two unnamed migrations\. Pass name= to each",
                ):
                    NamespaceAwareSerde(
                        namespaces=[Reshaped], migrations=[first, second]
                    )

        def with_two_stacked_decorators():
            def it_is_rejected_at_decoration_time():
                # Same treatment as a duplicate origin on ``@migrate_from``:
                # refuse where the duplication happens, not at construction
                # with a message naming one migration twice.
                with pytest.raises(ValueError, match="one transform per class"):

                    class Stacked(Namespace):
                        @transform_fields(_int_count)
                        @transform_fields(_drop_legacy_flag)
                        class Persisted(DomainEvent):
                            note: str = ""

    def when_the_transform_is_not_callable():
        def it_is_rejected_at_serde_construction():
            from langgraph_events.serde.migrations import Migration

            broken = Migration.transform_fields(
                target=Reshaped.Trimmed, transform="drop_legacy_flag"
            )

            with pytest.raises(TypeError, match=r"Reshaped\.Trimmed.*callable"):
                NamespaceAwareSerde(namespaces=[Reshaped], migrations=[broken])

    def when_the_target_is_unresolvable():
        def it_is_rejected_at_serde_construction():
            # Same rule as AddField: the target resolves live, or it is a
            # rename source.
            from langgraph_events.serde.migrations import Migration

            ghost = Migration.transform_fields(
                module="ghost.mod", qualname="Ghost.Gone", transform=_drop_legacy_flag
            )

            with pytest.raises(ValueError, match="TransformFields target"):
                NamespaceAwareSerde(migrations=[ghost])

    def when_a_transform_and_a_fill_share_one_stage():
        def it_runs_the_transform_first_and_the_fill_never_overwrites():
            from langgraph_events.serde.migrations import (
                AddField,
                Migration,
                TransformFields,
            )

            module = Reshaped.__module__

            def note_from_flag(kw: dict[str, Any]) -> dict[str, Any]:
                if kw.pop("legacy_flag", None):
                    kw["note"] = "from-transform"
                return kw

            serde = NamespaceAwareSerde(
                namespaces=[Reshaped],
                migrations=[
                    Migration(
                        name="transform-then-fill",
                        operations=(
                            TransformFields(module, "Reshaped.Trimmed", note_from_flag),
                            AddField(module, "Reshaped.Trimmed", "note", "from-fill"),
                        ),
                    )
                ],
            )

            def revive(**kwargs: Any) -> Any:
                return serde.loads_typed(
                    synthesize_legacy_payload(module, "Reshaped.Trimmed", kwargs)
                )

            assert revive(legacy_flag=True).note == "from-transform"
            assert revive().note == "from-fill"

    def when_one_decorator_carries_a_transform_and_a_backfill():
        def it_runs_the_transform_then_the_origin_fill():
            serde = NamespaceAwareSerde(namespaces=[DecoOriginBoth])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    DecoOriginBoth.__module__,
                    "DecoOriginBoth.Old",
                    {"legacy_flag": True},
                )
            )

            assert revived == DecoOriginBoth.Persisted(source="old")

    def when_legacy_write_meets_a_transform():
        # A transform has no inverse, so a payload written under the
        # historic identity would not carry the shape old pods expect.

        def with_a_class_stage_transform():
            def it_is_rejected_at_serde_construction():
                with pytest.raises(ValueError, match="legacy_write"):
                    NamespaceAwareSerde(namespaces=[Reshaped], legacy_write=True)

        def with_an_origin_stage_transform():
            def it_is_rejected_at_serde_construction():
                with pytest.raises(ValueError, match="legacy_write"):
                    NamespaceAwareSerde(
                        namespaces=[DecoOriginTransform], legacy_write=True
                    )


def _upper_legacy_flag(kw: dict[str, Any]) -> dict[str, Any]:
    """Raises ``AttributeError`` on the gate's ``None`` placeholder."""
    kw["note"] = kw.pop("legacy_flag").upper()
    return kw


class GateTrimmed(Namespace):
    """Dropped ``legacy_flag``, with a transform that reads its value."""

    @transform_fields(_upper_legacy_flag)
    class Persisted(DomainEvent):
        note: str = ""


def describe_TransformFields_revive_gate():
    # A v3 baseline records a field the live class dropped. The gate sends
    # a ``None`` placeholder for it, so the transform is exercised through
    # the real read path.

    def _baseline(tmp_path: Any, namespace: type) -> Any:
        return _baseline_v3_file(
            tmp_path,
            events=(
                (
                    namespace.__module__,
                    f"{namespace.__name__}.Persisted",
                    ["legacy_flag", "note"],
                ),
            ),
        )

    def when_the_baseline_records_a_dropped_field():
        def without_a_transform():
            def it_fails_pointing_at_TransformFields(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[DecoOriginTransform])

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_revive(
                        serde, _baseline(tmp_path, DecoOriginTransform)
                    )

                message = str(excinfo.value)
                assert "The baseline recorded 'legacy_flag'." in message
                assert "@transform_fields" in message

        def with_a_transform_that_drops_it():
            def it_passes(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    Migration,
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(
                    namespaces=[DecoOriginTransform],
                    migrations=[
                        Migration.transform_fields(
                            target=DecoOriginTransform.Persisted,
                            transform=_drop_legacy_flag,
                        )
                    ],
                )

                assert_all_baselined_revive(
                    serde, _baseline(tmp_path, DecoOriginTransform)
                )

        def with_a_transform_that_rejects_the_placeholder():
            def it_fails_naming_the_placeholder_contract(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    assert_all_baselined_revive,
                )

                serde = NamespaceAwareSerde(namespaces=[GateTrimmed])

                with pytest.raises(AssertionError) as excinfo:
                    assert_all_baselined_revive(serde, _baseline(tmp_path, GateTrimmed))

                message = str(excinfo.value)
                assert "TransformFields raised AttributeError" in message
                assert "None placeholder" in message
                assert "synthesize_legacy_payload" in message


# Fixtures for ``SplitEvent``. ``Work.Completed`` stays live and keeps its
# stored identity. A payload whose ``result`` says error builds
# ``Work.Failed`` instead (#125). ``result`` is a plain dict, so the
# payload needs no msgpack allowlist entry.
def _select_failed(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None
    return Work.Failed, {"reason": result["message"]}


class Work(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    class Completed(DomainEvent):
        result: dict[str, Any] = dataclasses.field(default_factory=dict)


_ERROR = {"status": "error", "message": "boom"}
_OK = {"status": "ok", "message": ""}


def _outcome_to_result(kw: dict[str, Any]) -> dict[str, Any]:
    """Class-stage transform: ``outcome`` was renamed to ``result``."""
    if "outcome" in kw:
        kw["result"] = kw.pop("outcome")
    return kw


def _select_staged_failed(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None
    return WorkStaged.Failed, {"reason": result["message"]}


# The source carries a rename, a class-stage transform and a class fill.
# The target carries its own class fill. The read order is: rename,
# transform, split, then the fills of the RESULTING identity only. A
# source fill applied after a split would reach ``Failed``, which has no
# ``attempt`` field, and fail the read.
class WorkStaged(Namespace):
    @backfill("severity", default="high")
    class Failed(DomainEvent):
        severity: str
        reason: str = ""

    @backfill("attempt", default=1)
    @transform_fields(_outcome_to_result)
    @migrate_from("WorkStaged.OldCompleted")
    class Completed(DomainEvent):
        attempt: int
        result: dict[str, Any] = dataclasses.field(default_factory=dict)


# The decorator form, collected from ``namespaces=`` like ``@transform_fields``.
# ``select_failed`` and ``Job`` are the example in docs/event-migrations.md,
# "Splitting one stored event into two on a payload value". Keep them in
# step: ``describe_SplitEvent_docs_example`` runs the docs snippet verbatim.
def select_failed(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None  # keep Job.Completed
    return Job.Failed, {"reason": result["message"]}  # resolved at read time


class Job(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(select_failed, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any] = dataclasses.field(default_factory=dict)


def _select_gate_failed(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None
    return GateSplit.Failed, {"reason": result["message"]}


def _select_reading_placeholder(kw: dict[str, Any]) -> tuple[type, dict[str, Any]]:
    """Raises ``TypeError`` on the gate's ``None`` placeholder."""
    return GateSplitStrict.Failed, {"reason": kw["result"]["message"]}


# ``result`` is required, so the revive gate sends a ``None`` placeholder
# for it and the select sees it.
class GateSplit(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(_select_gate_failed, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any]


class GateSplitStrict(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(_select_reading_placeholder, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any]


def describe_SplitEvent():
    # One stored identity, two live classes. ``select`` owns the access
    # to the discriminating value and picks the target (#125).

    def _serde(**kwargs: Any) -> NamespaceAwareSerde:
        from langgraph_events.serde.migrations import Migration, SplitEvent

        return NamespaceAwareSerde(
            namespaces=[Work],
            migrations=[
                Migration(
                    name="split-failed",
                    operations=(
                        SplitEvent(
                            module=Work.__module__,
                            qualname="Work.Completed",
                            select=_select_failed,
                            targets=(Work.Failed,),
                        ),
                    ),
                )
            ],
            **kwargs,
        )

    def _revive(serde: NamespaceAwareSerde, qualname: str, **kwargs: Any) -> Any:
        return serde.loads_typed(
            synthesize_legacy_payload(Work.__module__, qualname, kwargs)
        )

    def when_select_returns_a_target():
        def it_builds_the_target_from_the_returned_kwargs():
            revived = _revive(_serde(), "Work.Completed", result=_ERROR)

            assert revived == Work.Failed(reason="boom")

    def when_select_returns_None():
        def it_keeps_the_stored_class_and_kwargs():
            revived = _revive(_serde(), "Work.Completed", result=_OK)

            assert revived == Work.Completed(result=_OK)

    def when_the_current_release_wrote_the_source():
        def it_round_trips_unchanged():
            serde = _serde()
            stored = Work.Completed(result=_OK)

            assert serde.loads_typed(serde.dumps_typed(stored)) == stored

    def when_the_source_carries_a_transform_and_both_classes_carry_a_fill():
        def _staged_serde() -> NamespaceAwareSerde:
            from langgraph_events.serde.migrations import Migration, SplitEvent

            return NamespaceAwareSerde(
                namespaces=[WorkStaged],
                migrations=[
                    Migration(
                        operations=(
                            SplitEvent(
                                module=WorkStaged.__module__,
                                qualname="WorkStaged.Completed",
                                select=_select_staged_failed,
                                targets=(WorkStaged.Failed,),
                            ),
                        )
                    )
                ],
            )

        def _revive_staged(qualname: str, **kwargs: Any) -> Any:
            return _staged_serde().loads_typed(
                synthesize_legacy_payload(WorkStaged.__module__, qualname, kwargs)
            )

        def it_runs_the_transform_then_the_split_then_the_target_fills():
            revived = _revive_staged("WorkStaged.Completed", outcome=_ERROR)

            assert revived == WorkStaged.Failed(severity="high", reason="boom")

        def it_runs_the_source_fills_on_a_payload_select_keeps():
            revived = _revive_staged("WorkStaged.Completed", outcome=_OK)

            assert revived == WorkStaged.Completed(attempt=1, result=_OK)

        def it_splits_a_payload_stored_under_a_historic_name():
            revived = _revive_staged("WorkStaged.OldCompleted", outcome=_ERROR)

            assert revived == WorkStaged.Failed(severity="high", reason="boom")

    def _broken_serde(select: Any) -> NamespaceAwareSerde:
        from langgraph_events.serde.migrations import Migration, SplitEvent

        return NamespaceAwareSerde(
            namespaces=[Work],
            migrations=[
                Migration(
                    operations=(
                        SplitEvent(
                            module=Work.__module__,
                            qualname="Work.Completed",
                            select=select,
                            targets=(Work.Failed,),
                        ),
                    )
                )
            ],
        )

    _stored = synthesize_legacy_payload(
        Work.__module__, "Work.Completed", {"result": _ERROR}
    )
    _prefix = f"Cannot revive {Work.__module__}.Work.Completed: SplitEvent raised "

    def when_select_raises():
        # ormsgpack swallows an exception raised inside the ext-hook and
        # re-raises a bare ``ext_hook failed``. The split runs inside the
        # hook, so its failure must reach the ``errors`` channel, the
        # same one a TransformFields failure uses.

        def _raising(kw: dict[str, Any]) -> Any:
            return Work.Failed, {"reason": kw["missing"]}

        def it_raises_naming_the_stored_identity_and_the_cause():
            with pytest.raises(ValueError) as excinfo:
                _broken_serde(_raising).loads_typed(_stored)

            message = str(excinfo.value)
            assert message.startswith(_prefix + "KeyError: 'missing'.")
            assert "return None" in message

        def with_tolerate_unresolved():
            def it_degrades_and_collects_the_stored_identity():
                serde = _broken_serde(_raising)
                with serde.tolerate_unresolved() as unresolved:
                    revived = serde.loads_typed(_stored)

                placeholder = UnrevivedIdentity(Work.__module__, "Work.Completed")
                assert revived == placeholder
                assert unresolved == [placeholder]

    def when_select_returns_the_wrong_shape():
        # ``targets`` was validated at construction. A target outside it
        # would build a class the serde never validated, so the read
        # refuses it like every other bad return value.

        def _message(select: Any) -> str:
            with pytest.raises(ValueError) as excinfo:
                _broken_serde(select).loads_typed(_stored)
            return str(excinfo.value)

        def it_refuses_a_target_outside_targets():
            message = _message(lambda kw: (Work.Completed, kw))

            assert message.startswith(
                _prefix + "ValueError: select returned Work.Completed, which is "
                "not in targets="
            )

        def it_refuses_a_return_value_that_is_not_a_tuple():
            message = _message(lambda kw: {"reason": "boom"})

            assert message.startswith(
                _prefix + "TypeError: select returned dict, expected None or a "
                "(target, kwargs) tuple"
            )

        def it_refuses_kwargs_that_are_not_a_dict():
            message = _message(lambda kw: (Work.Failed, ["boom"]))

            assert message.startswith(
                _prefix + "TypeError: select returned kwargs of type list, "
                "expected dict"
            )

    def when_the_declaration_is_wrong():
        # Every refusal here would otherwise surface on the first
        # production read. Same strictness as TransformFields.

        def _split(name: str = "bad-split", **overrides: Any) -> Any:
            from langgraph_events.serde.migrations import Migration, SplitEvent

            fields: dict[str, Any] = {
                "module": Work.__module__,
                "qualname": "Work.Completed",
                "select": _select_failed,
                "targets": (Work.Failed,),
            }
            fields.update(overrides)
            return Migration(name=name, operations=(SplitEvent(**fields),))

        def it_refuses_a_source_that_is_a_rename_source():
            # The rename table cannot host a split: the source stays live,
            # and a rename source never resolves live.
            keyed_on_origin = _split(
                module=WorkStaged.__module__,
                qualname="WorkStaged.OldCompleted",
                targets=(WorkStaged.Failed,),
            )

            with pytest.raises(ValueError, match="split on the live target"):
                NamespaceAwareSerde(
                    namespaces=[WorkStaged], migrations=[keyed_on_origin]
                )

        def it_refuses_a_source_that_does_not_resolve():
            ghost = _split(module="ghost.mod", qualname="Ghost.Gone")

            with pytest.raises(ValueError, match=r"SplitEvent source ghost\.mod"):
                NamespaceAwareSerde(namespaces=[Work], migrations=[ghost])

        def it_refuses_a_target_that_is_not_an_Event():
            not_an_event = _split(targets=(dict,))

            with pytest.raises(ValueError, match="SplitEvent target dict"):
                NamespaceAwareSerde(namespaces=[Work], migrations=[not_an_event])

        def it_refuses_a_target_that_does_not_resolve():
            # A class nested in a function carries ``<locals>`` in its
            # qualname. Outside ``namespaces=`` nothing reaches it.
            class Local(Namespace):
                class Unreachable(DomainEvent):
                    reason: str = ""

            unreachable = _split(targets=(Local.Unreachable,))

            with pytest.raises(ValueError, match=r"SplitEvent target .*Unreachable"):
                NamespaceAwareSerde(namespaces=[Work], migrations=[unreachable])

        def it_refuses_empty_targets():
            # Refused where the op is built, before any serde sees it.
            with pytest.raises(ValueError, match=r"targets.*at least one"):
                _split(targets=())

        def it_refuses_a_select_that_is_not_callable():
            not_callable = _split(select="select_failed")

            with pytest.raises(TypeError, match=r"Work\.Completed.*callable"):
                NamespaceAwareSerde(namespaces=[Work], migrations=[not_callable])

        def it_refuses_two_splits_on_one_identity():
            # The read path runs one split per identity. A second one
            # would be silently ignored. Compose the cases in one select.
            with pytest.raises(ValueError, match="Duplicate SplitEvent"):
                NamespaceAwareSerde(
                    namespaces=[Work], migrations=[_split(), _split(name="again")]
                )

        def it_refuses_legacy_write():
            # A split has no inverse, the same rule as a transform.
            with pytest.raises(ValueError, match=r"legacy_write.*SplitEvent"):
                NamespaceAwareSerde(
                    namespaces=[Work], migrations=[_split()], legacy_write=True
                )

    def when_targets_is_a_list():
        def it_normalises_to_a_tuple_and_stays_hashable():
            # The decorator, the sugar and the raw op all pass through one
            # normalisation, so a list author gets the same op as a tuple
            # author, and the frozen op stays usable as a dict key.
            from langgraph_events.serde.migrations import Migration, SplitEvent

            raw = SplitEvent(
                module=Work.__module__,
                qualname="Work.Completed",
                select=_select_failed,
                targets=[Work.Failed],
            )
            sugared = Migration.split_event(
                source=Work.Completed, select=_select_failed, targets=[Work.Failed]
            ).operations[0]

            assert raw.targets == (Work.Failed,)
            assert sugared == raw
            assert hash(raw) == hash(sugared)

    def when_the_source_carries_the_decorator():
        def it_is_collected_from_namespaces():
            serde = NamespaceAwareSerde(namespaces=[Job])

            revived = serde.loads_typed(
                synthesize_legacy_payload(
                    Job.__module__, "Job.Completed", {"result": _ERROR}
                )
            )

            assert revived == Job.Failed(reason="boom")

        def with_two_stacked_decorators():
            def it_is_rejected_at_decoration_time():
                from langgraph_events.serde import split_event

                with pytest.raises(ValueError, match="one split per class"):

                    class Stacked(Namespace):
                        class Failed(DomainEvent):
                            reason: str = ""

                        @split_event(_select_failed, targets=(Failed,))
                        @split_event(_select_failed, targets=(Failed,))
                        class Completed(DomainEvent):
                            result: dict[str, Any] = dataclasses.field(
                                default_factory=dict
                            )

    def when_authored_through_the_sugar():
        def it_splits_like_the_raw_operation():
            from langgraph_events.serde import Migration

            serde = NamespaceAwareSerde(
                namespaces=[Work],
                migrations=[
                    Migration.split_event(
                        source=Work.Completed,
                        select=_select_failed,
                        targets=(Work.Failed,),
                    )
                ],
            )

            assert _revive(serde, "Work.Completed", result=_ERROR) == Work.Failed(
                reason="boom"
            )


def describe_SplitEvent_revive_gate():
    # The gate sends a ``None`` placeholder for a required field. A select
    # must return None on it, so the gate proves the source still
    # constructs. Each branch is pinned with real values through
    # ``synthesize_legacy_payload`` instead.

    def when_select_returns_None_on_the_placeholder():
        def it_passes(tmp_path: Any):
            from langgraph_events.serde.migrations import assert_all_baselined_revive

            baseline = _baseline_v3_file(
                tmp_path,
                events=((GateSplit.__module__, "GateSplit.Completed", ["result"]),),
            )

            assert_all_baselined_revive(
                NamespaceAwareSerde(namespaces=[GateSplit]), baseline
            )

    def when_select_reads_the_placeholder():
        def it_fails_naming_the_placeholder_contract(tmp_path: Any):
            from langgraph_events.serde.migrations import assert_all_baselined_revive

            baseline = _baseline_v3_file(
                tmp_path,
                events=(
                    (
                        GateSplitStrict.__module__,
                        "GateSplitStrict.Completed",
                        ["result"],
                    ),
                ),
            )

            with pytest.raises(AssertionError) as excinfo:
                assert_all_baselined_revive(
                    NamespaceAwareSerde(namespaces=[GateSplitStrict]), baseline
                )

            message = str(excinfo.value)
            assert "SplitEvent raised TypeError" in message
            assert "None placeholder" in message
            assert "synthesize_legacy_payload" in message


def describe_SplitEvent_docs_example():
    # The snippet under "Splitting one stored event into two on a payload
    # value" in docs/event-migrations.md, verbatim.

    def it_pins_both_branches():
        serde = NamespaceAwareSerde(namespaces=[Job])

        failed = serde.loads_typed(
            synthesize_legacy_payload(
                Job.__module__,
                "Job.Completed",
                {"result": {"status": "error", "message": "disk full"}},
            )
        )
        assert failed == Job.Failed(reason="disk full")

        completed = serde.loads_typed(
            synthesize_legacy_payload(
                Job.__module__, "Job.Completed", {"result": {"status": "ok"}}
            )
        )
        assert completed == Job.Completed(result={"status": "ok"})
