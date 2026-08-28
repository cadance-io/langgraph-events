"""Several independent engine lifetimes in one process (issue #148).

Namespace names identify a namespace within a *graph*, not within a process,
so a second lifetime may redefine a name the first one used. These tests are
the end-to-end proof: a lifetime runs, checkpoints, ends, and a fresh lifetime
resumes the same log against freshly-defined classes.
"""

import importlib
import re

import _lifetime_namespaces
import pytest
from _lifetime_namespaces import Filled as _Filled
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    IntegrationEvent,
    Namespace,
    on,
)
from langgraph_events._reducer import _matches_namespace
from langgraph_events.serde import NamespaceAwareSerde


def _next_lifetime():
    """End the current lifetime and start a new one, as a redeploy would."""
    importlib.reload(_lifetime_namespaces)
    return _lifetime_namespaces.Trading


def _make_trading():
    """A `Trading` namespace unrelated to any other `Trading`."""

    class Trading(Namespace):
        class Place(Command):
            sym: str

            class Placed(DomainEvent):
                sym: str

    return Trading


def _labelled_pair(message: str) -> tuple[str, str]:
    """The `X and Y` pair a collision diagnostic names, as two strings."""
    match = re.search(r"[: ]([^:]+?) and (.+?)\.", message)
    assert match is not None, message
    return match.group(1), match.group(2)


def describe_sequential_lifetimes():

    def when_a_second_lifetime_redefines_the_same_namespace_name():

        def it_does_not_raise():
            first = _next_lifetime()
            second = _next_lifetime()

            assert first is not second
            assert first.__namespace_name__ == second.__namespace_name__ == "Trading"

        def it_gives_each_graph_its_own_namespace():
            first = _next_lifetime()
            graph_one = EventGraph([first.Place])
            second = _next_lifetime()
            graph_two = EventGraph([second.Place])

            assert graph_one._namespaces["Trading"] is first
            assert graph_two._namespaces["Trading"] is second

        def it_discovers_reducers_from_its_own_lifetime():
            first = _next_lifetime()
            graph_one = EventGraph([first.Place])
            second = _next_lifetime()
            graph_two = EventGraph([second.Place])

            # Same reducer *name* either side; the objects must not be shared.
            assert graph_one._reducers["last_symbol"] is first.last_symbol
            assert graph_two._reducers["last_symbol"] is second.last_symbol
            assert first.last_symbol is not second.last_symbol

    def when_the_second_lifetime_resumes_the_first_lifetimes_checkpoint():

        def it_revives_the_log_against_the_new_classes():
            saver = MemorySaver()
            config = {"configurable": {"thread_id": "lifetimes"}}

            first = _next_lifetime()
            saver.serde = NamespaceAwareSerde(namespaces=[first])
            EventGraph([first.Place], checkpointer=saver).invoke(
                first.Place(sym="AAPL"), config=config
            )

            second = _next_lifetime()
            saver.serde = NamespaceAwareSerde(namespaces=[second])
            graph_two = EventGraph([second.Place], checkpointer=saver)

            revived = graph_two.get_state(config).events.latest(second.Place.Placed)

            assert revived.sym == "AAPL"
            assert type(revived) is second.Place.Placed
            assert type(revived) is not first.Place.Placed

    def when_an_earlier_lifetimes_serde_reads_its_own_checkpoint():

        # Both lifetimes render as the same ``(module, qualname)``, so an
        # import walk hands lifetime one's serde lifetime two's classes —
        # a silent cross-lifetime bleed. The serde resolves through its own
        # ``namespaces=`` scope first, which the two do not share (#150).
        def it_revives_into_that_lifetimes_classes():
            first = _next_lifetime()
            serde_one = NamespaceAwareSerde(namespaces=[first])
            blob = serde_one.dumps_typed(first.Place.Placed(sym="AAPL"))

            second = _next_lifetime()

            revived = serde_one.loads_typed(blob)

            assert type(revived) is first.Place.Placed
            assert type(revived) is not second.Place.Placed

    def when_reflection_queries_the_second_lifetime():

        def it_resolves_names_to_that_lifetimes_classes():
            first = _next_lifetime()
            EventGraph([first.Place]).namespaces()
            second = _next_lifetime()

            model = EventGraph([second.Place]).namespaces()

            assert model.namespaces["Trading"].cls is second
            assert model.namespaces["Trading"].commands["Place"].cls is second.Place


def describe_namespaces_reaching_one_graph():

    # Within a single graph the name must identify one class: reducer
    # discovery and reflection both group by it.
    def when_two_different_namespaces_share_a_name():

        def it_raises_naming_both_classes():
            one, two = _make_trading(), _make_trading()

            @on(one.Place)
            def handle_one(event: object) -> None:
                return None

            @on(two.Place)
            def handle_two(event: object) -> None:
                return None

            with pytest.raises(
                TypeError, match=r"Two different namespaces named 'Trading'"
            ):
                EventGraph([handle_one, handle_two])

    def when_a_produced_type_belongs_to_another_lifetime():

        # Subscribed types alone are not enough: a handler can subscribe to
        # one lifetime and *return* another's class, which merges the two
        # silently in the model and makes reflection answer from the wrong
        # lifetime.
        def it_raises():
            first = _next_lifetime()
            second = _next_lifetime()

            @on(first.Noted)
            def react(event: object) -> second.Place:
                return second.Place(sym="AAPL")

            with pytest.raises(
                TypeError, match=r"Two different namespaces named 'Trading'"
            ):
                EventGraph([react])

    def when_both_definitions_come_from_one_module():

        # Two lifetimes of one module render identically as
        # module.qualname, so the message has to say more than that or it
        # names the same string twice and explains nothing.
        def it_does_not_name_the_same_string_twice():
            first = _next_lifetime()
            second = _next_lifetime()

            @on(first.Place)
            def handle_one(event: object) -> None:
                return None

            @on(second.Place)
            def handle_two(event: object) -> None:
                return None

            with pytest.raises(TypeError) as exc:
                EventGraph([handle_one, handle_two])

            here, there = _labelled_pair(str(exc.value))
            assert here != there

    def when_two_handlers_share_one_namespace():

        def it_does_not_raise():
            one = _make_trading()

            @on(one.Place)
            def handle_a(event: object) -> None:
                return None

            @on(one.Place.Placed)
            def handle_b(event: object) -> None:
                return None

            assert EventGraph([handle_a, handle_b])._namespaces["Trading"] is one


def describe_namespace_scoped_reducers():

    # A declarative reducer on a Namespace folds only that namespace's
    # events. Membership is the namespace *object*, not its name — with
    # names no longer unique process-wide, a name match would let one
    # lifetime's reducer fold another's event. Tested at the predicate
    # because the graph-build guards make an end-to-end repro unreachable.
    def when_the_event_belongs_to_another_lifetime():

        def it_does_not_match():
            first = _next_lifetime()
            second = _next_lifetime()

            assert not _matches_namespace(first.Place.Placed(sym="AAPL"), second)

    def when_the_event_belongs_to_that_namespace():

        def it_matches():
            first = _next_lifetime()

            assert _matches_namespace(first.Place.Placed(sym="AAPL"), first)


def describe_a_serde_given_more_than_one_lifetime():

    # The scope map is keyed by (module, qualname), which two lifetimes of one
    # module share. Binding last-wins would make revival depend on the order of
    # a sequence that reads as insignificant. EventGraph rejects the same
    # mistake; so should the serde.
    def when_two_namespaces_contribute_one_identity():

        def it_raises_rather_than_binding_silently():
            first = _next_lifetime()
            second = _next_lifetime()

            with pytest.raises(ValueError, match=r"same event identity") as exc:
                NamespaceAwareSerde(namespaces=[first, second])

            # Both classes render identically — sharing (module, qualname) is
            # the trigger — so whatever the message names them by, the two
            # halves must differ rather than printing one string twice.
            here, there = _labelled_pair(str(exc.value))
            assert here != there

    def when_the_same_namespace_is_passed_twice():

        def it_is_accepted():
            first = _next_lifetime()

            assert NamespaceAwareSerde(namespaces=[first, first]) is not None


def describe_events_outside_a_namespace():

    # Module-level IntegrationEvents are not reached by the namespace walk,
    # so before #155 they resolved by import and bled across lifetimes.
    def when_the_serde_is_given_them_explicitly():

        def it_keeps_each_lifetimes_own_class():
            first = _next_lifetime()
            first_ping = _lifetime_namespaces.Ping
            serde = NamespaceAwareSerde(namespaces=[first], events=[first_ping])
            blob = serde.dumps_typed({"e": first_ping(sym="AAPL")})

            second = _next_lifetime()

            revived = serde.loads_typed(blob)["e"]
            assert type(revived) is first_ping
            assert type(revived) is not _lifetime_namespaces.Ping
            assert second is not first

    def when_a_decorator_records_history_on_one():

        def it_collects_the_migration(self=None):
            from langgraph_events.serde.migrations import migrate_from

            @migrate_from("Ancient")
            class Loose(IntegrationEvent):
                sym: str

            serde = NamespaceAwareSerde(namespaces=[], events=[Loose])

            assert serde._rename_table  # decorator on a loose event is honoured


def describe_from_namespaces_auto_wiring():

    # The auto-wired serde should cover every event the graph touches, not
    # just the namespaced ones — the caller passed namespaces, not an event
    # inventory, so anything else has to be derived.
    def when_a_handler_produces_an_event_outside_the_namespaces():

        def it_scopes_that_event_too():
            from langgraph.checkpoint.memory import MemorySaver

            first = _next_lifetime()
            ping = _lifetime_namespaces.Ping

            @on(first.Place.Placed)
            def echo(event: object) -> _lifetime_namespaces.Ping:
                return ping(sym="AAPL")

            graph = EventGraph.from_namespaces(
                first, handlers=[echo], checkpointer=MemorySaver()
            )

            ids = graph._checkpointer.serde.revivable_identities()
            assert (ping.__module__, "Ping") in ids

        def it_scopes_namespaces_reached_only_through_handlers():
            # A namespace can arrive via handlers= rather than namespaces=.
            # Passing only the declared domains would leave its events
            # resolving by import — the bleed this all exists to close.
            from langgraph.checkpoint.memory import MemorySaver

            first = _next_lifetime()
            audited = _lifetime_namespaces.Audited

            @on(first.Place.Placed)
            def log(event: object) -> _lifetime_namespaces.Audited.Logged:
                return audited.Logged(sym="AAPL")

            graph = EventGraph.from_namespaces(
                first, handlers=[log], checkpointer=MemorySaver()
            )

            ids = graph._checkpointer.serde.revivable_identities()
            assert (audited.__module__, "Audited.Logged") in ids

        def it_does_not_consume_the_namespace_model_cache():
            # Deriving the loose events from a NamespaceModel would fire its
            # design-smell warnings from inside the library and populate the
            # cache, so a user's later graph.namespaces() would never warn.
            from langgraph.checkpoint.memory import MemorySaver

            first = _next_lifetime()
            ping = _lifetime_namespaces.Ping

            @on(first.Place.Placed)
            def echo(event: object) -> _lifetime_namespaces.Ping:
                return ping(sym="AAPL")

            graph = EventGraph.from_namespaces(
                first, handlers=[echo], checkpointer=MemorySaver()
            )

            assert graph._namespaces_cache is None


def describe_one_class_reached_twice():

    # namespaces= and events= can name the same class. Collecting it twice
    # would collect its @migrate_from twice, and a duplicated AddField is
    # rejected by _flatten_and_validate as a double fill.
    def when_a_nested_event_is_also_passed_directly():

        def it_is_collected_once():
            # Collecting twice would declare the same fill twice, which
            # _flatten_and_validate rejects as a duplicate AddField.
            serde = NamespaceAwareSerde(
                namespaces=[_Filled],
                events=[_Filled.Do.Done],
            )

            assert serde._origin_addfield_table


class _Alef(IntegrationEvent):
    sym: str


class _Bet(IntegrationEvent):
    sym: str


def describe_loose_event_ordering():

    # The loose events come out of set arithmetic, whose iteration order
    # varies per process. That order reaches migration collection and decides
    # which class a collision diagnostic names first, so a message must not
    # change between runs of the same code.
    def when_a_graph_touches_several():

        def it_returns_them_sorted_and_deduplicated():
            @on(_Alef)
            def one(event: _Alef) -> _Bet:
                return _Bet(sym=event.sym)

            @on(_Bet)
            def two(event: _Bet) -> None:
                return None

            loose = EventGraph([one, two])._loose_events

            assert list(loose) == sorted(
                set(loose), key=lambda c: (c.__module__, c.__qualname__)
            )
            assert len(loose) == len(set(loose))
            assert set(loose) == {_Alef, _Bet}
