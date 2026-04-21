"""Tests for runtime return-type enforcement.

The framework enforces handler return contracts at dispatch time:
- If the handler subscribes to a ``Command``, it must return one of the
  command's nested outcomes (``Command.Outcomes``) or ``None``.
- If the handler has a return annotation, the returned value must be an
  instance of the declared union.
- Otherwise (no annotation, not a Command subscription), falls back to
  the shape-only check (``Event | None | Scatter``).
"""

from __future__ import annotations

import pytest
from conftest import Order

from langgraph_events import (
    Aggregate,
    Command,
    DomainEvent,
    EventGraph,
    IntegrationEvent,
    Scatter,
    on,
)


class Alien(Aggregate):
    class Do(Command):
        class Stuff(DomainEvent):
            pass


class Ping(IntegrationEvent):
    pass


class Pong(IntegrationEvent):
    pass


class Foreign(IntegrationEvent):
    pass


class Batch(IntegrationEvent):
    pass


class Item(IntegrationEvent):
    pass


def describe_return_contract():

    def describe_command_subscription():

        def when_return_is_a_declared_outcome():

            def it_accepts():
                @on(Order.Place)
                def place(event: Order.Place) -> Order.Place.Placed:
                    return Order.Place.Placed(order_id="o1")

                graph = EventGraph([place])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(Order.Place.Placed)

        def when_return_is_not_a_declared_outcome():

            def it_raises_TypeError():
                @on(Order.Place)
                def place(event: Order.Place):
                    return Alien.Do.Stuff()  # foreign outcome

                graph = EventGraph([place])
                with pytest.raises(
                    TypeError, match=r"return.*Order\.Place|must return"
                ):
                    graph.invoke(Order.Place(customer_id="c1"))

        def when_return_is_None():

            def it_accepts():
                @on(Order.Place)
                def place(event: Order.Place) -> None:
                    return None

                graph = EventGraph([place])
                # No violation — None is always allowed.
                graph.invoke(Order.Place(customer_id="c1"))

        def when_handler_annotated_none_returns_event():

            def it_raises():
                @on(Order.Place)
                def cheat(event: Order.Place) -> None:
                    return Alien.Do.Stuff()  # type: ignore[return-value]

                graph = EventGraph([cheat])
                with pytest.raises(TypeError, match=r"permits only None"):
                    graph.invoke(Order.Place(customer_id="c1"))

    def describe_annotated_non_command_handler():

        def when_return_matches_annotation():

            def it_accepts():
                @on(Ping)
                def handle(event: Ping) -> Pong:
                    return Pong()

                graph = EventGraph([handle])
                log = graph.invoke(Ping())
                assert log.has(Pong)

        def when_return_violates_annotation():

            def it_raises_TypeError():
                @on(Ping)
                def handle(event: Ping) -> Pong:
                    return Foreign()  # type: ignore[return-value]

                graph = EventGraph([handle])
                with pytest.raises(TypeError, match=r"return|must return"):
                    graph.invoke(Ping())

    def describe_unannotated_non_command_handler():

        def when_handler_returns_any_Event():

            def it_accepts():
                @on(Ping)
                def handle(event):
                    return Pong()

                graph = EventGraph([handle])
                log = graph.invoke(Ping())
                assert log.has(Pong)

    def describe_scatter_returns():

        def when_scatter_element_matches_declared_type():

            def it_accepts():
                @on(Batch)
                def split(event: Batch) -> Scatter[Item]:
                    return Scatter([Item(), Item()])

                @on(Item)
                def take(event: Item) -> None:
                    return None

                graph = EventGraph([split, take])
                log = graph.invoke(Batch())
                assert log.count(Item) == 2

        def when_scatter_element_violates_declared_type():

            def it_raises_TypeError():
                @on(Batch)
                def split(event: Batch) -> Scatter[Item]:
                    return Scatter([Item(), Foreign()])  # type: ignore[list-item]

                @on(Item)
                def take(event: Item) -> None:
                    return None

                graph = EventGraph([split, take])
                with pytest.raises(TypeError, match=r"Scatter|scattered"):
                    graph.invoke(Batch())
