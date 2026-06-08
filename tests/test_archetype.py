"""Lifecycle archetype composition: inherit an archetype command's behavior
while keeping per-entity identity + typing, with a Policy bridge (issue #98)."""

import pytest

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Namespace,
    namespace_of,
)


class Lifecyclic(Namespace):
    """Archetype: defines the lifecycle behavior once. The handler emits its
    outcome reflectively (``type(self).Persisted``) so subclasses emit their
    own per-entity event, and carries no return annotation so each subclass's
    ``Outcomes`` drives the contract."""

    class Persist(Command):
        value: str = ""

        def handle(self):
            return type(self).Persisted(out=self.value)


class Sprocket(Namespace):
    class Persist(Lifecyclic.Persist):
        class Persisted(DomainEvent):
            out: str = ""


class Cog(Namespace):
    class Persist(Lifecyclic.Persist):
        class Persisted(DomainEvent):
            out: str = ""


class Openable(Namespace):
    """Archetype whose shared handler delegates per-entity work to the
    consuming namespace's ``Policy`` (the bridge)."""

    class Open(Command):
        subject: str = ""

        def handle(self):
            policy = namespace_of(self).Policy()
            return type(self).Opened(ref=policy.open(self.subject))


class Account(Namespace):
    class Open(Openable.Open):
        class Opened(DomainEvent):
            ref: str = ""

    class Policy:
        def open(self, subject: str) -> str:
            return f"account::{subject}"


class Ticket(Namespace):
    class Open(Openable.Open):
        class Opened(DomainEvent):
            ref: str = ""

    class Policy:
        def open(self, subject: str) -> str:
            return f"ticket::{subject}"


class Overrider(Namespace):
    class Persist(Lifecyclic.Persist):
        class Overridden(DomainEvent):
            note: str = ""

        def handle(self):
            return type(self).Overridden(note="custom")


def describe_archetype_command_inheritance():

    def when_two_commands_subclass_one_archetype_command():

        def it_gives_each_a_distinct_handler():
            assert (
                Sprocket.Persist.__command_handler__
                is not Cog.Persist.__command_handler__
            )

        def without_a_handler_collision():

            def it_builds_one_graph():
                graph = EventGraph([Sprocket.Persist, Cog.Persist])
                assert {"Sprocket.Persist", "Cog.Persist"} <= set(graph.handler_names)


def describe_per_entity_outcome():

    def when_subclasses_define_their_own_event():

        def it_keeps_distinct_event_identity():
            assert Sprocket.Persist.Persisted is not Cog.Persist.Persisted

        def it_synthesizes_per_entity_outcomes():
            assert Sprocket.Persist.Outcomes is Sprocket.Persist.Persisted

        def it_stamps_the_consuming_namespace():
            assert Sprocket.Persist.Persisted.__namespace__ == "Sprocket"

    def when_an_inherited_command_is_invoked():

        def it_emits_its_own_event_reflectively():
            log = EventGraph([Sprocket.Persist, Cog.Persist]).invoke(
                Sprocket.Persist(value="hi")
            )
            assert log.latest(Sprocket.Persist.Persisted) == Sprocket.Persist.Persisted(
                out="hi"
            )
            assert log.latest(Cog.Persist.Persisted) is None


def describe_namespace_of():
    """Bridge enabler: an archetype handler resolves the consuming namespace
    (and its ``Policy``) from the command instance."""

    def when_given_a_command_class():

        def it_returns_the_owning_namespace():
            assert namespace_of(Sprocket.Persist) is Sprocket

    def when_given_an_event_instance():

        def it_returns_the_owning_namespace():
            assert namespace_of(Cog.Persist(value="x")) is Cog


def describe_policy_bridge():

    def when_two_domains_compose_one_archetype():

        def it_emits_each_domains_own_event():
            graph = EventGraph([Account.Open, Ticket.Open])
            log = graph.invoke(Account.Open(subject="alice"))
            assert log.latest(Account.Open.Opened) is not None
            assert log.latest(Ticket.Open.Opened) is None

        def it_dispatches_to_each_domains_policy():
            graph = EventGraph([Account.Open, Ticket.Open])
            opened_a = graph.invoke(Account.Open(subject="alice")).latest(
                Account.Open.Opened
            )
            opened_t = graph.invoke(Ticket.Open(subject="bug")).latest(
                Ticket.Open.Opened
            )
            assert opened_a.ref == "account::alice"
            assert opened_t.ref == "ticket::bug"


def describe_handler_override():

    def when_a_subclass_defines_its_own_handler():

        def it_uses_the_overriding_handler():
            log = EventGraph([Overrider.Persist]).invoke(Overrider.Persist(value="x"))
            assert log.latest(Overrider.Persist.Overridden) is not None


def describe_uses_declaration():

    def when_archetypes_are_declared():

        def it_records_them_for_introspection():
            class Vault(Namespace, uses=[Openable]):
                class Open(Openable.Open):
                    class Opened(DomainEvent):
                        ref: str = ""

                class Policy:
                    def open(self, subject: str) -> str:
                        return subject

            assert Vault.__uses__ == (Openable,)

    def when_no_archetypes_are_declared():

        def it_defaults_to_empty():
            class Plain(Namespace):
                pass

            assert Plain.__uses__ == ()

    def when_a_non_namespace_is_declared():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Namespace"):

                class BadUses(Namespace, uses=[object]):
                    pass
