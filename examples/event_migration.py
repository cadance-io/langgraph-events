"""Event migration smoke example.

Shows the end-to-end loop for migrating renamed event classes:

1. Define the event under its **current** location, with ``@migrate_from``
   recording the historic qualname it used to live at.
2. Build the graph with ``EventGraph.from_namespaces`` and a checkpointer.
   That is the *entire* authoring story — ``from_namespaces`` auto-wires a
   namespace-scoped migration serde, so you never construct
   ``NamespaceAwareSerde`` or repeat a namespace tuple.
3. Synthesize a checkpoint payload as a prior library version would have
   produced it and show it revives at the current class.

The advanced ``legacy_write=True`` rolling-deploy path is shown last: it
is opt-in via an explicitly-constructed serde, which also demonstrates the
auto-wiring opt-out (a user-supplied serde always wins).

Usage::

    uv run python examples/event_migration.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Namespace
from langgraph_events.serde import NamespaceAwareSerde, synthesize_legacy_payload
from langgraph_events.serde.migrations import migrate_from


class Persona(Namespace):
    """Cadance's persona namespace, mid-refactor.

    `Persona.Persisted` used to be a sibling of `Persona.Persist`;
    it's now nested inside the producing Command. The decorator records
    the historic qualname so old checkpoints still revive.
    """

    class Persist(Command):
        note: str = ""

        @migrate_from("Persona.Persisted")
        class Persisted(DomainEvent):
            note: str = ""

        def handle(self) -> Persona.Persist.Persisted:
            return Persona.Persist.Persisted(note=self.note)


def main() -> None:
    # The whole authoring story: @migrate_from on the class (above) plus a
    # checkpointer. from_namespaces scopes a migration-aware serde to these
    # namespaces for you — no NamespaceAwareSerde import, no namespace
    # tuple threaded into a second place.
    graph = EventGraph.from_namespaces(Persona, checkpointer=MemorySaver())

    # The graph wired this serde; shown here only to exercise the read
    # path directly. Normally the graph uses it internally on every
    # checkpoint read, so a legacy payload "just works" on the next run.
    serde = graph._checkpointer.serde
    assert isinstance(serde, NamespaceAwareSerde)

    legacy = synthesize_legacy_payload(
        Persona.__module__, "Persona.Persisted", {"note": "from-old-release"}
    )
    revived = serde.loads_typed(legacy)
    print(f"Revived legacy payload as: {type(revived).__qualname__}")
    print(f"  note = {revived.note!r}")
    assert isinstance(revived, Persona.Persist.Persisted)

    # Advanced / opt-in: rolling deploys. Construct the serde yourself with
    # legacy_write=True so new pods encode under the OLDEST qualname and
    # old pods (previous release) can still revive during the rollout. A
    # user-supplied serde also wins over auto-wiring (the opt-out).
    rolling = MemorySaver(
        serde=NamespaceAwareSerde(namespaces=(Persona,), legacy_write=True)
    )
    rolling_graph = EventGraph.from_namespaces(Persona, checkpointer=rolling)
    assert rolling_graph._checkpointer.serde is rolling.serde

    fresh = Persona.Persist.Persisted(note="rolling-deploy")
    _, bytes_under_old_name = rolling.serde.dumps_typed(fresh)
    assert b"Persona.Persisted" in bytes_under_old_name
    assert b"Persona.Persist.Persisted" not in bytes_under_old_name
    print(
        "\nlegacy_write=True encodes 'Persona.Persisted' instead of "
        "'Persona.Persist.Persisted' — old pods will revive it."
    )


if __name__ == "__main__":
    main()
