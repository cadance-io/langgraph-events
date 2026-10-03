"""Tests for the reducer view of a NamespaceModel and the focused mermaid diagram."""

from __future__ import annotations

import json
import re
from typing import ClassVar, Protocol, runtime_checkable

import pytest

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    Invariant,
    InvariantViolated,
    Namespace,
    NamespaceModel,
    ScalarReducer,
    on,
)
from langgraph_events._namespace._mermaid import _LINKSTYLE_MUTED


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


class _Unlocked(Invariant):
    pass


class _Gate(Namespace):
    class Open(Command):
        invariants: ClassVar = {_Unlocked: lambda log: False}

        class Opened(DomainEvent):
            pass

        def handle(self) -> _Gate.Open.Opened:
            return _Gate.Open.Opened()


@on(_Clock.Tick.Ticked)
def request_open(event: _Clock.Tick.Ticked) -> _Gate.Open:
    return _Gate.Open()


@on(InvariantViolated, invariant=_Unlocked)
def explain_locked(event: InvariantViolated) -> _Ops.Ping:
    return _Ops.Ping()


@on(HandlerRaised)
def report_failure(event: HandlerRaised) -> _Ops.Ping:
    return _Ops.Ping()


@runtime_checkable
class _HasEdge(Protocol):
    edge: float


proto_total = ScalarReducer(
    name="proto_total", event_type=_HasEdge, fn=lambda e: e.edge
)


@on(_Ops.Ping.Pinged)
def commits(event: _Ops.Ping.Pinged) -> None:
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


Focus = NamespaceModel.Focus


def _focused(**selected: tuple[str, ...]) -> str:
    return _model().mermaid(focus=Focus(**selected))


def describe_show_raises():
    def when_it_is_true():
        def it_draws_the_raises_edges():
            assert "(raises)" in _model().mermaid()

    def when_it_is_false():
        def it_draws_no_raises_edge():
            assert "(raises)" not in _model().mermaid(show_raises=False)

        def it_draws_no_node_that_only_a_raises_edge_reaches():
            output = _model().mermaid(show_raises=False)
            assert "HandlerRaised([HandlerRaised])" not in output


def describe_focus():
    def when_it_selects_nothing():
        def it_raises():
            with pytest.raises(ValueError, match="selects nothing"):
                Focus()

    def when_a_field_is_one_string():
        def it_reads_the_string_as_one_name():
            assert Focus(namespaces="_Ledger") == Focus(namespaces=("_Ledger",))

    def when_a_field_is_a_set():
        def it_compares_equal_whatever_the_order():
            assert Focus(namespaces={"b", "a"}) == Focus(namespaces=("a", "b"))

    def when_a_field_is_bytes():
        def it_raises_type_error():
            with pytest.raises(TypeError, match="names must be str"):
                Focus(namespaces=b"_Ledger")

    def when_it_names_an_unknown_reaction():
        def it_names_the_nearest_valid_reaction_in_the_error():
            with pytest.raises(ValueError, match="note_each_tick"):
                _focused(reactions=("note_each_tik",))

    def when_it_selects_a_namespace():
        def it_draws_no_entry_arrow():
            assert "==>" not in _focused(namespaces=("_Ledger",))

        def it_draws_the_commands_and_outcomes_of_the_namespace():
            assert "Commit --> Committed" in _focused(namespaces=("_Ledger",))

        def it_leaves_out_an_unrelated_namespace():
            assert "_Ops" not in _focused(namespaces=("_Ledger",))

        def it_draws_a_reducer_of_the_namespace_as_selected():
            output = _focused(namespaces=("_Ledger",))
            assert "_reducer_commits[(commits)]:::reducer" in output

        def it_draws_a_reducer_outside_the_namespace_as_context():
            output = _focused(namespaces=("_Ledger",))
            assert "_reducer_edge_total[(edge_total)]:::ctx" in output

    def when_it_selects_a_reaction():
        def it_draws_the_edges_of_the_reaction():
            output = _focused(reactions=("note_each_tick",))
            assert 'Ticked ==>|"note_each_tick [orchestrate]"| Note' in output

        def it_draws_the_endpoints_as_context():
            output = _focused(reactions=("note_each_tick",))
            assert "Ticked(Ticked):::ctx" in output
            assert "Note{{Note}}:::ctx" in output

        def it_titles_the_namespace_of_a_context_node_as_context():
            output = _focused(reactions=("note_each_tick",))
            assert '_Clock["_Clock namespace (context)"]' in output

        def it_leaves_out_the_outcome_of_a_context_command():
            assert "Noted" not in _focused(reactions=("note_each_tick",))

    def when_it_selects_a_reducer():
        def it_draws_the_events_the_reducer_folds():
            output = _focused(reducers=("edge_total",))
            assert "Committed -.->|folds| _reducer_edge_total" in output


def _noted(**notes: str) -> str:
    return _model().mermaid(notes=notes)


def describe_notes():
    def when_a_note_names_a_command():
        def it_adds_the_note_as_a_second_line():
            assert 'Commit{{"Commit<br>7 records"}}' in _noted(
                **{"_Ledger.Commit": "7 records"}
            )

    def when_a_note_names_a_reducer():
        def it_adds_the_note_as_a_second_line():
            output = _noted(edge_total="sum edge")
            assert '_reducer_edge_total[("edge_total<br>sum edge")]' in output

    def when_the_note_has_two_lines():
        def it_draws_each_line_under_the_name():
            output = _noted(edge_total="sum edge\n= 0.4")
            assert '_reducer_edge_total[("edge_total<br>sum edge<br>= 0.4")]' in output

    def when_the_note_holds_markup():
        def it_writes_each_special_character_as_an_entity():
            output = _noted(**{"_Ledger.Commit": '<b>"x"</b> #1'})
            assert "#lt;b#gt;#quot;x#quot;#lt;/b#gt; #35;1" in output
            assert "<b>" not in output

    def when_the_note_holds_a_directive():
        def it_writes_each_percent_sign_as_an_entity():
            output = _noted(**{"_Ledger.Commit": '%%{init: {"theme":"dark"}}%%'})
            assert (
                "#37;#37;{init: {#quot;theme#quot;:#quot;dark#quot;}}#37;#37;" in output
            )

    def when_a_note_names_an_unknown_node():
        def it_names_the_nearest_valid_node_in_the_error():
            with pytest.raises(ValueError, match=r"_Ledger\.Commit"):
                _noted(**{"_Ledger.Comit": "x"})

    def when_a_note_is_not_a_string():
        def it_writes_its_text():
            assert 'Commit{{"Commit<br>7"}}' in _noted(**{"_Ledger.Commit": 7})

    def when_a_note_is_empty():
        def it_draws_the_name_alone():
            output = _noted(**{"_Ledger.Commit": ""})

            assert "Commit{{Commit}}" in output
            assert "Commit<br>" not in output

    def when_a_note_key_is_wrong():
        def it_lists_the_valid_keys():
            with pytest.raises(ValueError, match=r"Valid keys: .*edge_total"):
                _noted(Flag="x")


def describe_muted():
    def when_it_names_a_command():
        def it_adds_the_muted_class_to_the_node():
            output = _model().mermaid(muted=("_Ledger.Commit",))
            assert "class Commit muted" in output

    def when_it_names_a_reducer():
        def it_adds_the_muted_class_to_the_reducer():
            output = _model().mermaid(muted=("edge_total",))
            assert "class _reducer_edge_total muted" in output

    def when_it_names_an_unknown_node():
        def it_names_the_nearest_valid_node_in_the_error():
            with pytest.raises(ValueError, match="edge_total"):
                _model().mermaid(muted=("edge_totl",))

    def when_it_is_one_string():
        def it_reads_the_string_as_one_name():
            output = _model().mermaid(muted="_Ledger.Commit")

            assert "class Commit muted" in output

    def when_it_is_empty():
        def it_declares_no_muted_class():
            assert "muted" not in _model().mermaid()


def _inline_name(command: type) -> str:
    return next(
        r.name
        for r in _model().command_handlers
        if r.inline and r.commands[0] is command
    )


def describe_reaction_keys():
    def when_a_note_names_a_reaction():
        def it_writes_the_note_under_its_edge_label():
            output = _model().mermaid(notes={"note_each_tick": "fired 2x"})

            assert (
                'Ticked -->|"note_each_tick [orchestrate]<br>fired 2x"| Note' in output
            )

    def when_muted_names_a_reaction():
        def it_fades_the_edges_of_the_reaction():
            output = _model().mermaid(muted=["note_each_tick"])
            edges = [
                line
                for line in output.splitlines()
                if re.search(r" (==>|-\.->|-\.-|-->)", line)
            ]
            index = edges.index('    Ticked -->|"note_each_tick [orchestrate]"| Note')

            assert f"linkStyle {index} {_LINKSTYLE_MUTED}" in output

    def when_a_name_is_a_reaction_and_a_reducer():
        def it_raises_value_error():
            model = EventGraph(
                [_Clock.Tick, _Ledger.Commit, note_each_tick, recover_ledger, commits]
            ).namespaces()

            with pytest.raises(ValueError, match="both a reaction and a node"):
                model.mermaid(notes={"commits": "x"})

    def when_a_key_is_an_inline_handler_name():
        def it_names_the_command_to_use_instead():
            name = _inline_name(_Ledger.Commit)

            with pytest.raises(ValueError, match=r"_Ledger\.Commit"):
                _model().mermaid(notes={name: "x"})

    def when_a_focus_selects_the_reaction():
        def it_draws_its_edges_as_thick_arrows():
            output = _focused(reactions=("note_each_tick",))

            assert 'Ticked ==>|"note_each_tick [orchestrate]"| Note' in output


def _gate_model() -> NamespaceModel:
    return EventGraph(
        [_Clock.Tick, request_open, _Gate.Open, explain_locked, _Ops.Ping]
    ).namespaces()


def describe_truth_of_the_drawing():
    def when_a_reducer_type_is_a_data_protocol():
        def it_states_that_its_folds_are_unknown():
            model = EventGraph(
                [_Ledger.Commit, recover_ledger], reducers=[proto_total]
            ).namespaces()
            reducer = next(r for r in model.reducers if r.name == "proto_total")

            assert reducer.subscribes is None
            assert "  proto_total  (folds unknown" in model.text()
            assert "proto_total<br>folds: unknown" in model.mermaid()

    def when_a_focus_reaches_an_invariant():
        def it_draws_the_invariant_as_context():
            output = _gate_model().mermaid(focus=Focus(reactions="explain_locked"))

            assert "_Unlocked{_Unlocked}:::ctx" in output

        def it_writes_a_note_on_the_invariant():
            qualname = _Unlocked.__qualname__
            output = _gate_model().mermaid(notes={qualname: "violated 2x"})

            assert '_Unlocked{"_Unlocked<br>violated 2x"}:::inv' in output

    def when_a_focus_selects_a_reducer_only():
        def it_does_not_title_its_box_as_context():
            output = _focused(reducers=("commits",))

            assert '_Ledger["_Ledger namespace"]' in output

    def when_raises_edges_are_hidden_but_the_event_stays():
        def it_says_that_its_producers_are_hidden():
            model = EventGraph([_Ledger.Commit, report_failure, _Ops.Ping]).namespaces()

            output = model.mermaid(show_raises=False)

            assert "HandlerRaised<br>raises edges hidden" in output


def describe_node_styles():
    def when_a_focus_mutes_a_context_node():
        def it_draws_muted_apart_from_context():
            output = _model().mermaid(
                focus=Focus(reactions="note_each_tick"), muted="_Clock.Note"
            )
            styles = {
                line.split()[1]: line.split(maxsplit=2)[2]
                for line in output.splitlines()
                if line.strip().startswith("classDef")
            }

            assert styles["muted"] != styles["ctx"]
            assert "font-style:italic" in styles["muted"]
