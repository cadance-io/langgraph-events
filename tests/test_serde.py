"""Tests for ``langgraph_events.serde`` — namespace-aware checkpoint serde."""

from __future__ import annotations

import warnings

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Namespace
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
