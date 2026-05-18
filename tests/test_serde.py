"""Tests for ``langgraph_events.serde`` — namespace-aware checkpoint serde."""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from typing import Any

import ormsgpack
import pytest
from conftest import Started
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt
from pydantic import BaseModel

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
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
    _option,
)
from langgraph_events.serde.migrations import backfill, migrate_from


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
@dataclasses.dataclass
class PlainPayload:
    name: str
    count: int


# Pydantic v2 model — separate ext-code branch (``EXT_PYDANTIC_V2``) than
# ``PlainPayload``. Pydantic-heavy state was the reported repro for #68.
class PydanticPayload(BaseModel):
    name: str
    count: int


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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="rename-persona-approved",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
                    ),
                    Migration(
                        name="rename-story-approved",
                        operations=(
                            RenameEvent(
                                old_module="legacy.story",
                                old_qualname="Legacy.StoryApproved",
                                new_module=Story.__module__,
                                new_qualname="Story.Approve.Approved",
                            ),
                        ),
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="rename-legacy-persona",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
                    ),
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="release-N",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.A",
                                new_module="legacy.persona",
                                new_qualname="Legacy.B",
                            ),
                        ),
                    ),
                    Migration(
                        name="release-N+1",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.B",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
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
                    from langgraph_events.serde.migrations import (
                        AddField,
                        Migration,
                    )

                    migrations = [
                        Migration(
                            name="add-command-id",
                            operations=(
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
                    payload = synthesize_legacy_payload(
                        NestedPersist.__module__,
                        "NestedPersist.Persist.Persisted",
                        {"note": "old"},
                    )

                    revived = serde.loads_typed(payload)

                    assert isinstance(revived, NestedPersist.Persist.Persisted)
                    assert revived.command_id == "legacy"
                    assert revived.note == "old"

            def when_default_factory_is_used_for_mutable_value():
                def it_invokes_the_factory_per_payload():
                    # Two distinct payloads must NOT share the same list
                    # object — that would be a classic mutable-default bug
                    # made worse by living one layer down inside the serde.
                    from langgraph_events.serde.migrations import (
                        AddField,
                        Migration,
                    )

                    migrations = [
                        Migration(
                            name="add-tags",
                            operations=(
                                AddField(
                                    module=NestedPersist.__module__,
                                    qualname="NestedPersist.Persist.Persisted",
                                    field="tags",
                                    default_factory=list,
                                ),
                            ),
                        ),
                    ]
                    # Override the default tuple field with a list-typed one
                    # by going through kwargs directly — we want to observe
                    # whether revival hands back the SAME list instance.
                    serde = NamespaceAwareSerde(migrations=migrations)
                    raw1 = synthesize_legacy_payload(
                        NestedPersist.__module__,
                        "NestedPersist.Persist.Persisted",
                        {"command_id": "c1", "note": "a"},
                    )
                    raw2 = synthesize_legacy_payload(
                        NestedPersist.__module__,
                        "NestedPersist.Persist.Persisted",
                        {"command_id": "c2", "note": "b"},
                    )

                    r1 = serde.loads_typed(raw1)
                    r2 = serde.loads_typed(raw2)

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
                    from langgraph_events.serde.migrations import (
                        AddField,
                        Migration,
                    )

                    migrations = [
                        Migration(
                            name="add-command-id",
                            operations=(
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
                    payload = synthesize_legacy_payload(
                        NestedPersist.__module__,
                        "NestedPersist.Persist.Persisted",
                        {"command_id": "from-old", "note": "n"},
                    )

                    revived = serde.loads_typed(payload)

                    assert revived.command_id == "from-old"

        def when_event_was_renamed():
            def it_revives_under_new_qualname():
                # Simulate a checkpoint produced by an older build where the
                # event class lived at "legacy.module:Legacy.Approved", later
                # renamed to the live `Persona.Approve.Approved`.
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                payload = synthesize_legacy_payload(
                    module="legacy.persona",
                    qualname="Legacy.Approved",
                    kwargs={"note": "hi"},
                )
                migrations = [
                    Migration(
                        name="rename-legacy-persona-approved",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
                    ),
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="rename-to-persona",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
                    ),
                    Migration(
                        name="rename-to-story",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Story.__module__,
                                new_qualname="Story.Approve.Approved",
                            ),
                        ),
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                # `Persona.Approve.Approved` IS live — declaring a migration
                # whose old name points at it would silently rewrite live
                # payloads on read.
                migrations = [
                    Migration(
                        name="shadows-live",
                        operations=(
                            RenameEvent(
                                old_module=Persona.__module__,
                                old_qualname="Persona.Approve.Approved",
                                new_module=Story.__module__,
                                new_qualname="Story.Approve.Approved",
                            ),
                        ),
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

                with pytest.raises(ValueError, match=r"does not resolve"):
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="forward",
                        operations=(
                            RenameEvent(
                                old_module="legacy.a",
                                old_qualname="A",
                                new_module="legacy.b",
                                new_qualname="B",
                            ),
                        ),
                    ),
                    Migration(
                        name="reverse",
                        operations=(
                            RenameEvent(
                                old_module="legacy.b",
                                old_qualname="B",
                                new_module="legacy.a",
                                new_qualname="A",
                            ),
                        ),
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
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                migrations = [
                    Migration(
                        name="self",
                        operations=(
                            RenameEvent(
                                old_module=Persona.__module__,
                                old_qualname="Persona.Approve.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
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

    def describe_revivable_identities():
        # The set of ``(module, qualname)`` this serde will accept on a
        # read: live classes in scope, plus every historic identity a
        # rename migration can rewrite (decorator-driven AND hand-authored,
        # unioned — users want one question answered, "will this revive?",
        # not "live or migrated?").

        def it_returns_every_revivable_identity():
            from langgraph_events.serde.migrations import (
                Migration,
                RenameEvent,
            )

            serde = NamespaceAwareSerde(
                namespaces=[DecoReorg],
                migrations=[
                    Migration(
                        name="legacy-rename",
                        operations=(
                            RenameEvent(
                                old_module="legacy.persona",
                                old_qualname="Legacy.Approved",
                                new_module=Persona.__module__,
                                new_qualname="Persona.Approve.Approved",
                            ),
                        ),
                    ),
                ],
            )

            ids = serde.revivable_identities()

            # Live class reached by the namespace walk.
            assert (DecoReorg.__module__, "DecoReorg.Persist.Persisted") in ids
            # Decorator-driven historic source.
            assert (DecoReorg.__module__, "DecoReorg.Persisted") in ids
            # Hand-authored historic source.
            assert ("legacy.persona", "Legacy.Approved") in ids

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

    def describe_assert_covers():
        # The CI gate: every identity the previous release wrote must be
        # either still live in this serde's namespaces or covered by a
        # rename migration, else the next production read of that payload
        # would fail. Raising is the right verb — it's a pytest assertion.

        def when_baseline_identity_has_no_migration():
            def it_raises_naming_the_uncovered_identity(tmp_path: Any):
                from langgraph_events.serde.migrations.detect import (
                    MigrationCoverageError,
                )

                serde = NamespaceAwareSerde()
                baseline = _baseline_file(tmp_path, ("ghost.mod", "Ghost.Gone"))

                with pytest.raises(MigrationCoverageError) as excinfo:
                    serde.assert_covers(baseline)

                assert "Ghost.Gone" in str(excinfo.value)
                assert excinfo.value.uncovered == (("ghost.mod", "Ghost.Gone"),)

        def when_baseline_identity_has_a_decorator_migration():
            def it_passes_silently(tmp_path: Any):
                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path, (DecoReorg.__module__, "DecoReorg.Persisted")
                )

                serde.assert_covers(baseline)

        def when_baseline_identity_is_still_live():
            def it_passes_silently(tmp_path: Any):
                serde = NamespaceAwareSerde(namespaces=[DecoReorg])
                baseline = _baseline_file(
                    tmp_path,
                    (DecoReorg.__module__, "DecoReorg.Persist.Persisted"),
                )

                serde.assert_covers(baseline)

        def when_baseline_identity_has_a_hand_authored_migration():
            def it_passes_silently(tmp_path: Any):
                from langgraph_events.serde.migrations import (
                    Migration,
                    RenameEvent,
                )

                serde = NamespaceAwareSerde(
                    migrations=[
                        Migration(
                            name="legacy-rename",
                            operations=(
                                RenameEvent(
                                    old_module="legacy.persona",
                                    old_qualname="Legacy.Approved",
                                    new_module=Persona.__module__,
                                    new_qualname="Persona.Approve.Approved",
                                ),
                            ),
                        ),
                    ],
                )
                baseline = _baseline_file(
                    tmp_path, ("legacy.persona", "Legacy.Approved")
                )

                serde.assert_covers(baseline)

    def describe_assert_all_baselined_revive():
        # Stronger than assert_covers: instead of a set-membership check,
        # it pushes a synthesized legacy payload for every baselined
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

        def it_still_exposes_the_decorator_and_sugar_tier():
            import langgraph_events.serde as serde_pkg

            for name in (
                "NamespaceAwareSerde",
                "Migration",
                "migrate_from",
                "synthesize_legacy_payload",
                "assert_all_baselined_revive",
            ):
                assert hasattr(serde_pkg, name), name

    def when_the_composite_escape_hatch_is_needed():
        def it_keeps_raw_ops_importable_from_serde_migrations():
            from langgraph_events.serde.migrations import AddField, RenameEvent

            assert RenameEvent is not None
            assert AddField is not None


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
