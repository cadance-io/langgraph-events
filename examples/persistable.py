"""Lifecycle archetype composition — langgraph-events demo (issue #98).

A reusable *archetype* defines a lifecycle's **behavior once**; consuming
domains inherit it while keeping their **own per-entity event identity and
field types**, and plug in per-entity operations through a **Policy** bridge.

This mirrors a real pattern: a persistence lifecycle (``Persist`` →
``Approve``) shared across several entity domains (Persona, Story, …) that each
carry a differently-typed candidate and persist to their own store.

Key pieces:

- ``Persistable`` — the archetype ``Namespace``. Its command handlers emit
  outcomes **reflectively** (``type(self).Persisted``) and delegate per-entity
  work to ``namespace_of(self).Policy`` — so one handler serves every domain.
- ``Persona`` / ``Story`` — domains that ``uses=[Persistable]`` and **subclass**
  the archetype commands to give them typed candidates + their own events, plus
  a ``Policy`` implementation. Subclassing an archetype command rebinds a
  distinct handler, so they coexist as graph nodes without collision.
- One reaction, ``auto_approve``, reacts to *every* domain's ``Persisted`` via
  ``namespace_of`` — the same bridge, at the reaction layer.

Usage:
    python examples/persistable.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Namespace,
    namespace_of,
    on,
)

# In-memory "database" the policies persist into, keyed by domain.
_STORE: dict[str, dict[str, object]] = {}


class Persistable(Namespace):
    """Archetype: the persist → approve lifecycle, defined once."""

    class Policy(Protocol):
        """The bridge: per-entity operations each consuming domain implements."""

        def persist(self, candidate: object) -> str: ...

        def approve(self, entity_id: str) -> None: ...

    class Persist(Command):
        candidate: object = None

        def handle(self):
            policy = namespace_of(self).Policy()
            return type(self).Persisted(entity_id=policy.persist(self.candidate))

    class Approve(Command):
        entity_id: str = ""

        def handle(self):
            namespace_of(self).Policy().approve(self.entity_id)
            return type(self).Approved(entity_id=self.entity_id)


@dataclass
class PersonaCandidate:
    name: str = ""


@dataclass
class StoryCandidate:
    title: str = ""


class Persona(Namespace, uses=[Persistable]):
    class Persist(Persistable.Persist):
        candidate: PersonaCandidate = field(default_factory=PersonaCandidate)

        class Persisted(DomainEvent):
            entity_id: str = ""

    class Approve(Persistable.Approve):
        class Approved(DomainEvent):
            entity_id: str = ""

    class Policy:
        def persist(self, candidate: object) -> str:
            assert isinstance(candidate, PersonaCandidate)
            entity_id = f"persona:{candidate.name}"
            _STORE.setdefault("persona", {})[entity_id] = candidate
            return entity_id

        def approve(self, entity_id: str) -> None:
            _STORE["persona"][entity_id] = "approved"


class Story(Namespace, uses=[Persistable]):
    class Persist(Persistable.Persist):
        candidate: StoryCandidate = field(default_factory=StoryCandidate)

        class Persisted(DomainEvent):
            entity_id: str = ""

    class Approve(Persistable.Approve):
        class Approved(DomainEvent):
            entity_id: str = ""

    class Policy:
        def persist(self, candidate: object) -> str:
            assert isinstance(candidate, StoryCandidate)
            entity_id = f"story:{candidate.title}"
            _STORE.setdefault("story", {})[entity_id] = candidate
            return entity_id

        def approve(self, entity_id: str) -> None:
            _STORE["story"][entity_id] = "approved"


@on(Persona.Persist.Persisted, Story.Persist.Persisted)
def auto_approve(
    event: Persona.Persist.Persisted | Story.Persist.Persisted,
) -> Persona.Approve | Story.Approve:
    """One reaction, every domain: resolve the domain via ``namespace_of`` and
    approve it. Returns that domain's own ``Approve`` command."""
    return namespace_of(event).Approve(entity_id=event.entity_id)


def main() -> None:
    graph = EventGraph(
        [Persona.Persist, Persona.Approve, Story.Persist, Story.Approve, auto_approve]
    )

    persona_log = graph.invoke(Persona.Persist(candidate=PersonaCandidate(name="ada")))
    story_log = graph.invoke(Story.Persist(candidate=StoryCandidate(title="quest")))

    # Each domain emits ITS OWN per-entity event (distinct identity), produced
    # by the single inherited archetype handler dispatching through its Policy.
    persisted = persona_log.latest(Persona.Persist.Persisted)
    approved = persona_log.latest(Persona.Approve.Approved)
    print(f"persona persisted -> {type(persisted).__qualname__}: {persisted.entity_id}")
    print(f"persona approved  -> {type(approved).__qualname__}: {approved.entity_id}")

    s_persisted = story_log.latest(Story.Persist.Persisted)
    s_approved = story_log.latest(Story.Approve.Approved)
    print(
        f"story persisted   -> {type(s_persisted).__qualname__}: {s_persisted.entity_id}"
    )
    print(
        f"story approved    -> {type(s_approved).__qualname__}: {s_approved.entity_id}"
    )

    print(f"Persona uses {[ns.__name__ for ns in Persona.__uses__]}")
    print(f"store: {_STORE}")


if __name__ == "__main__":
    main()
