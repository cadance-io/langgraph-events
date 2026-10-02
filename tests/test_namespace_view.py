"""Tests for the reducer view of a NamespaceModel and the focused mermaid diagram."""

from __future__ import annotations

import json
from typing import ClassVar

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    Namespace,
    NamespaceModel,
    ScalarReducer,
    on,
)


class _LedgerError(Exception):
    pass


class _Clock(Namespace):
    class Tick(Command):
        class Ticked(DomainEvent):
            pass

        def handle(self) -> _Clock.Tick.Ticked:
            return _Clock.Tick.Ticked()

    class Note(Command):
        class Noted(DomainEvent):
            pass

        def handle(self) -> _Clock.Note.Noted:
            return _Clock.Note.Noted()


class _Ledger(Namespace):
    class Commit(Command):
        raises: ClassVar = (_LedgerError,)

        class Committed(DomainEvent):
            edge: float = 0.0

        def handle(self) -> _Ledger.Commit.Committed:
            return _Ledger.Commit.Committed()

    commits = ScalarReducer(event_type=Commit.Committed, fn=lambda e: e.edge)


class _Ops(Namespace):
    class Ping(Command):
        class Pinged(DomainEvent):
            pass

        def handle(self) -> _Ops.Ping.Pinged:
            return _Ops.Ping.Pinged()

    pings = ScalarReducer(event_type=DomainEvent, fn=lambda e: 1)


@on(_Clock.Tick.Ticked)
def note_each_tick(event: _Clock.Tick.Ticked) -> _Clock.Note:
    return _Clock.Note()


@on(HandlerRaised)
def recover_ledger(event: HandlerRaised) -> None:
    return None


edge_total = ScalarReducer(
    name="edge_total", event_type=_Ledger.Commit.Committed, fn=lambda e: e.edge
)


def _model() -> NamespaceModel:
    return EventGraph(
        [
            _Clock.Tick,
            _Clock.Note,
            _Ledger.Commit,
            _Ops.Ping,
            note_each_tick,
            recover_ledger,
        ],
        reducers=[edge_total],
    ).namespaces()


def _reducer(name: str) -> NamespaceModel.Reducer:
    return next(r for r in _model().reducers if r.name == name)


def describe_reducers():
    def when_a_namespace_declares_the_reducer():
        def it_names_the_namespace():
            assert _reducer("commits") == NamespaceModel.Reducer(
                name="commits",
                subscribes=(_Ledger.Commit.Committed,),
                namespace="_Ledger",
            )

    def when_the_graph_receives_the_reducer():
        def it_names_no_namespace():
            assert _reducer("edge_total").namespace is None

    def when_the_event_type_is_a_base_class():
        def it_subscribes_the_concrete_events_of_its_namespace():
            assert _reducer("pings").subscribes == (_Ops.Ping.Pinged,)

    def describe_json():
        def it_encodes_each_reducer_by_qualname():
            encoded = json.loads(_model().json())["reducers"]
            assert {
                "name": "commits",
                "subscribes": ["_Ledger.Commit.Committed"],
                "namespace": "_Ledger",
            } in encoded

    def describe_text():
        def it_lists_the_folded_events_of_each_reducer():
            assert "  edge_total  (folds Committed)" in _model().text()

    def describe_mermaid():
        def it_draws_the_reducer_as_a_cylinder():
            assert "_reducer_edge_total[(edge_total)]:::reducer" in _model().mermaid()

        def it_draws_a_folds_edge_from_each_event_it_reads():
            assert "Committed -.->|folds| _reducer_edge_total" in _model().mermaid()

        def it_puts_a_namespaced_reducer_inside_its_namespace():
            output = _model().mermaid()
            start = output.index('subgraph _Ledger["_Ledger namespace"]')
            assert "_reducer_commits" in output[start : output.index("end", start)]
