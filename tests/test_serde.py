"""Tests for ``langgraph_events.serde`` — namespace-aware checkpoint serde."""

from __future__ import annotations

import dataclasses
import warnings

import pytest
from conftest import Started
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt

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
from langgraph_events.serde import NamespaceAwareSerde


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
        def when_msgpack_encoding_fails():
            def it_warns_before_falling_through_to_super():
                # An object neither the namespace-aware ``_default`` nor the
                # upstream default knows how to encode. The namespace-aware
                # path raises MsgpackEncodeError; we warn, then defer to the
                # parent (which then raises further). The point of the test
                # is that the warning fires before the parent path runs — so
                # users notice the namespace-aware identity scheme is not in
                # effect for this blob (collision-prone).
                from langgraph_events.serde import _jsonplus

                class Unencodable:
                    pass

                serde = NamespaceAwareSerde()
                with (
                    warnings.catch_warnings(record=True) as caught,
                    pytest.raises(TypeError),
                ):
                    warnings.simplefilter("always")
                    serde.dumps_typed(Unencodable())

                fallback_warnings = [
                    w
                    for w in caught
                    if "fall" in str(w.message).lower()
                    or "namespace" in str(w.message).lower()
                ]
                assert fallback_warnings, (
                    f"expected a fallback warning, got: "
                    f"{[str(w.message) for w in caught]}"
                )

                # Sanity: helper module is importable (catches an obvious
                # M4-style refactor regression).
                assert _jsonplus.NamespaceAwareSerde is NamespaceAwareSerde

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
