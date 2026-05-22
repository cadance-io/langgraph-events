"""Causal-role taxonomy on ``NamespaceModel`` edges.

``Edge.kind`` says *how* an edge was produced (return / scatter / raise /
framework). ``Edge.causation`` says *what causal role* it plays:

- ``intent``      — inline ``Command.handle()`` emits its own nested outcome
- ``react``       — an ``@on`` reactor emits a ``DomainEvent``
- ``orchestrate`` — an ``@on`` reactor emits a ``Command`` (a saga move)
- ``chain``       — inline ``Command.handle()`` emits another ``Command``
- ``None``        — ``raises`` / ``framework`` edges carry no domain causation
"""

from __future__ import annotations

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    Interrupted,
    Namespace,
    Resumed,
    Scatter,
    on,
)


class _CausShop(Namespace):
    class Place(Command):
        class Placed(DomainEvent):
            pass

        def handle(self) -> _CausShop.Place.Placed:
            return _CausShop.Place.Placed()

    class Ship(Command):
        class Shipped(DomainEvent):
            pass

        def handle(self) -> _CausShop.Ship.Shipped:
            return _CausShop.Ship.Shipped()

    class Noted(DomainEvent):
        pass


@on(_CausShop.Place.Placed)
def caus_react(event: _CausShop.Place.Placed) -> _CausShop.Noted:
    return _CausShop.Noted()


@on(_CausShop.Noted)
def caus_orch(event: _CausShop.Noted) -> _CausShop.Ship:
    return _CausShop.Ship()


class _CausChain(Namespace):
    class Start(Command):
        def handle(self) -> _CausChain.Finish:
            return _CausChain.Finish()

    class Finish(Command):
        class Finished(DomainEvent):
            pass

        def handle(self) -> _CausChain.Finish.Finished:
            return _CausChain.Finish.Finished()


class _CausRaise(Namespace):
    class Risky(Command):
        raises = (ValueError,)

        class Done(DomainEvent):
            pass

        def handle(self) -> _CausRaise.Risky.Done:
            return _CausRaise.Risky.Done()


@on(HandlerRaised, exception=ValueError)
def caus_catch(event: HandlerRaised) -> None: ...


class _CausPause(Namespace):
    class Wait(Command):
        def handle(self) -> Interrupted:
            return Interrupted(reason="x")


@on(Resumed)
def caus_after(event: Resumed) -> None: ...


class _CausScatter(Namespace):
    class Trigger(DomainEvent):
        pass

    class Work(Command):
        class Worked(DomainEvent):
            pass

        def handle(self) -> _CausScatter.Work.Worked:
            return _CausScatter.Work.Worked()

    class Logged(DomainEvent):
        pass


@on(_CausScatter.Trigger)
def caus_burst(
    event: _CausScatter.Trigger,
) -> Scatter[_CausScatter.Logged | _CausScatter.Work]:
    return Scatter([_CausScatter.Logged()])


def _edge(d, src, tgt):
    return next(
        e for e in d.edges if e.source.__name__ == src and e.target.__name__ == tgt
    )


def describe_causation():

    def describe_intent():

        def when_inline_handle_returns_its_own_domain_event():
            def it_tags_the_edge_intent():
                d = EventGraph([_CausShop.Place, _CausShop.Ship]).namespaces()
                assert _edge(d, "Place", "Placed").causation == "intent"

    def describe_react():

        def when_reactor_returns_a_domain_event():
            def it_tags_the_edge_react():
                d = EventGraph(
                    [_CausShop.Place, _CausShop.Ship, caus_react]
                ).namespaces()
                assert _edge(d, "Placed", "Noted").causation == "react"

    def describe_orchestrate():

        def when_reactor_returns_a_command():
            def it_tags_the_edge_orchestrate():
                d = EventGraph(
                    [_CausShop.Place, _CausShop.Ship, caus_react, caus_orch]
                ).namespaces()
                assert _edge(d, "Noted", "Ship").causation == "orchestrate"

    def describe_chain():

        def when_inline_handle_returns_a_sibling_command():
            def it_tags_the_edge_chain():
                d = EventGraph([_CausChain.Start, _CausChain.Finish]).namespaces()
                assert _edge(d, "Start", "Finish").causation == "chain"

    def describe_none():

        def when_edge_is_a_raises_edge():
            def it_has_no_causation():
                d = EventGraph([_CausRaise.Risky, caus_catch]).namespaces()
                assert _edge(d, "Risky", "HandlerRaised").causation is None

        def when_edge_is_the_framework_resume_edge():
            def it_has_no_causation():
                d = EventGraph([_CausPause.Wait, caus_after]).namespaces()
                assert _edge(d, "Interrupted", "Resumed").causation is None

    def describe_scatter():

        def when_scatter_targets_mix_command_and_domain_event():
            def it_tags_each_target_independently():
                d = EventGraph([_CausScatter.Work, caus_burst]).namespaces()
                assert _edge(d, "Trigger", "Logged").causation == "react"
                assert _edge(d, "Trigger", "Work").causation == "orchestrate"

    def describe_renderings():

        def when_rendered_as_json():
            def it_includes_causation_per_edge():
                d = EventGraph(
                    [_CausShop.Place, _CausShop.Ship, caus_react, caus_orch]
                ).namespaces()
                blob = d.to_dict()
                edges = blob["edges"]
                assert edges
                assert all("causation" in e for e in edges)
                orch = next(
                    e
                    for e in edges
                    if e["source"].endswith("Noted") and e["target"].endswith("Ship")
                )
                assert orch["causation"] == "orchestrate"

        def when_rendered_as_text():
            def it_tags_orchestrate_and_chain_rows():
                d = EventGraph(
                    [_CausShop.Place, _CausShop.Ship, caus_react, caus_orch]
                ).namespaces()
                txt = d.text()
                assert "[orchestrate]" in txt

        def when_rendered_as_mermaid():
            def it_styles_orchestrate_distinctly():
                d = EventGraph(
                    [_CausShop.Place, _CausShop.Ship, caus_react, caus_orch]
                ).namespaces()
                mer = d.mermaid()
                assert "orchestrate" in mer
