"""Several independent engine lifetimes in one process (issue #148).

Namespace names identify a namespace within a *graph*, not within a process,
so a second lifetime may redefine a name the first one used. These tests are
the end-to-end proof: a lifetime runs, checkpoints, ends, and a fresh lifetime
resumes the same log against freshly-defined classes.
"""

from __future__ import annotations

import importlib

import _lifetime_namespaces
import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Namespace, on
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
