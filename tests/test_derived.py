"""Derived reducers: a value computed from other reducers, never stored."""

from __future__ import annotations

from typing import Any

import pytest
from ag_ui.core import EventType, RunAgentInput
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    Derived,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Namespace,
    ReducerNotSetError,
    ScalarReducer,
    message_reducer,
    on,
)
from langgraph_events.agui import AGUIAdapter, FrontendStateMutated
from langgraph_events.stream import StreamFrame


class Placed(IntegrationEvent):
    pass


class Shipped(IntegrationEvent):
    pass


class Reported(IntegrationEvent):
    stage: str = ""


placed = ScalarReducer(name="placed", event_type=Placed, fn=lambda e: True)
shipped = ScalarReducer(name="shipped", event_type=Shipped, fn=lambda e: True)


def _stage_of(placed: bool | None, shipped: bool | None) -> str | None:
    if shipped:
        return "shipped"
    if placed:
        return "placed"
    return None


stage = Derived(name="stage", fn=_stage_of)


@on(Placed, Shipped)
def report(event: Placed | Shipped, stage: str | None) -> Reported:
    return Reported(stage=stage or "unset")


@on(Placed)
def ship(event: Placed) -> Shipped:
    return Shipped()


class Checked(IntegrationEvent):
    pass


@on(Checked)
def check(event: Checked, stage: str | None) -> Reported:
    return Reported(stage=stage or "unset")


class Asked2(Interrupted):
    pass


class Answered(IntegrationEvent):
    pass


@on(Placed)
def ask(event: Placed) -> Asked2:
    return Asked2()


@on(Answered)
def answered(event: Answered, stage: str | None) -> Reported:
    return Reported(stage=stage or "unset")


def _reports(log: Any) -> list[str]:
    return [e.stage for e in log.filter(Reported)]


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


class Parcel(Namespace):
    sent = ScalarReducer(event_type=DomainEvent, fn=lambda e: True)
    status = Derived(fn=lambda sent: "sent" if sent else "held")

    class Look(Command):
        class Looked(DomainEvent):
            status: str

        def handle(self, status: str) -> Parcel.Look.Looked:
            return Parcel.Look.Looked(status=status)


def describe_Derived():
    def when_a_handler_declares_it_by_name():
        def it_receives_the_value_of_its_inputs_at_node_start():
            graph = EventGraph([report], reducers=[placed, shipped, stage])

            log = graph.invoke(Placed())

            assert _reports(log) == ["placed"]

        def it_recomputes_the_value_after_an_input_changes():
            graph = EventGraph([report, ship], reducers=[placed, shipped, stage])

            log = graph.invoke(Placed())

            assert _reports(log) == ["placed", "shipped"]

        def it_reads_the_empty_default_of_an_unset_input():
            graph = EventGraph([check], reducers=[placed, shipped, stage])

            log = graph.invoke(Checked())

            assert _reports(log) == ["unset"]

        def when_the_annotation_rejects_none():
            def it_raises_reducer_not_set_error():
                @on(Checked)
                def strict(event: Checked, stage: str) -> None:
                    return None

                graph = EventGraph([strict], reducers=[placed, shipped, stage])

                with pytest.raises(
                    ReducerNotSetError, match=r"Derived 'stage' returned None"
                ):
                    graph.invoke(Checked())

        def when_its_function_raises():
            def it_raises_outside_the_raises_boundary_of_the_handler():
                def _broken(placed: bool | None) -> str:
                    raise ValueError("bad input")

                broken = Derived(name="broken", fn=_broken)

                @on(Reported, raises=ValueError)
                def guarded(event: Reported, broken: str) -> None:
                    return None

                @on(HandlerRaised, exception=ValueError)
                def recover(event: HandlerRaised) -> None:
                    return None

                graph = EventGraph([guarded, recover], reducers=[placed, broken])

                with pytest.raises(ValueError, match="bad input") as caught:
                    graph.invoke(Reported())

                assert caught.value.__notes__[0] == (
                    "Derived 'broken' raised while it computed its value."
                )

    def when_the_run_is_checkpointed():
        def it_does_not_write_the_value_to_the_checkpoint():
            graph = EventGraph(
                [report], reducers=[placed, shipped, stage], checkpointer=MemorySaver()
            )

            graph.invoke(Placed(), config=_config("t1"))

            values = graph.compiled.get_state(_config("t1")).values
            assert values["placed"] is True
            assert "stage" not in values

        def it_recomputes_the_value_in_a_later_run():
            graph = EventGraph(
                [check], reducers=[placed, shipped, stage], checkpointer=MemorySaver()
            )
            graph.invoke(Placed(), config=_config("t2"))

            log = graph.invoke(Checked(), config=_config("t2"))

            assert _reports(log) == ["placed"]

        def it_computes_the_value_for_a_resumed_handler():
            graph = EventGraph(
                [ask, answered],
                reducers=[placed, shipped, stage],
                checkpointer=MemorySaver(),
            )
            graph.invoke(Placed(), config=_config("t4"))

            log = graph.resume(Answered(), config=_config("t4"))

            assert _reports(log) == ["placed"]
            values = graph.compiled.get_state(_config("t4")).values
            assert "stage" not in values

        async def it_reads_the_checkpointed_inputs_in_an_async_stream():
            graph = EventGraph(
                [check], reducers=[placed, shipped, stage], checkpointer=MemorySaver()
            )
            await graph.ainvoke(Placed(), config=_config("t5"))

            frames = [
                f
                async for f in graph.astream_events(
                    Checked(),
                    include_reducers=["stage"],
                    include_custom_events=True,
                    config=_config("t5"),
                )
            ]

            assert [f.reducers for f in frames] == [
                {"stage": "placed"},
                {"stage": "placed"},
            ]

        def it_refuses_to_pre_seed_the_value():
            graph = EventGraph(
                [report], reducers=[placed, shipped, stage], checkpointer=MemorySaver()
            )

            with pytest.raises(ValueError, match=r"Derived name\(s\) \['stage'\]"):
                graph.pre_seed(_config("t3"), {"stage": "shipped"})

    def when_declared_on_a_namespace():
        def it_takes_the_attribute_name():
            graph = EventGraph([Parcel.Look])

            log = graph.invoke(Parcel.Look())

            assert [e.status for e in log.filter(Parcel.Look.Looked)] == ["held"]

    def when_the_graph_is_built():
        def it_rejects_an_input_that_names_no_reducer():
            typo = Derived(name="stage", fn=lambda placd: placd)

            with pytest.raises(TypeError, match=r"'placd'.*\['placed', 'shipped'\]"):
                EventGraph([report], reducers=[placed, shipped, typo])

        def it_rejects_an_input_that_names_another_derived():
            twice = Derived(name="twice", fn=lambda stage: stage)

            with pytest.raises(TypeError, match=r"'twice' reads the derived 'stage'"):
                EventGraph([report], reducers=[placed, shipped, stage, twice])

        def it_rejects_a_variadic_parameter():
            loose = Derived(name="loose", fn=lambda **values: values)

            with pytest.raises(TypeError, match=r"'loose'.*variadic"):
                EventGraph([report], reducers=[placed, shipped, loose])

        def it_rejects_a_name_that_a_reducer_holds():
            clash = Derived(name="placed", fn=lambda shipped: shipped)

            with pytest.raises(TypeError, match=r"'placed'.*reducer and a Derived"):
                EventGraph([report], reducers=[placed, shipped, clash])

        def it_rejects_a_reserved_state_field_name():
            reserved = Derived(name="events", fn=lambda placed: placed)

            with pytest.raises(ValueError, match="reserved state fields"):
                EventGraph([report], reducers=[placed, shipped, reserved])

    def when_a_stream_includes_reducers():
        def it_includes_a_derived_that_the_list_names():
            graph = EventGraph([ship], reducers=[placed, shipped, stage])

            frames = list(graph.stream_events(Placed(), include_reducers=["stage"]))

            assert all(isinstance(f, StreamFrame) for f in frames)
            assert [f.reducers for f in frames] == [
                {"stage": "placed"},
                {"stage": "shipped"},
            ]

        async def it_marks_the_derived_changed_on_an_input_change():
            graph = EventGraph([ship], reducers=[placed, shipped, stage])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(), include_reducers=["stage"], include_custom_events=True
                )
            ]

            assert [f.reducers for f in frames] == [
                {"stage": "placed"},
                {"stage": "shipped"},
            ]
            assert [f.changed_reducers for f in frames] == [
                frozenset({"stage"}),
                frozenset({"stage"}),
            ]

        def it_includes_every_derived_for_true():
            graph = EventGraph([ship], reducers=[placed, shipped, stage])

            frames = list(graph.stream_events(Placed(), include_reducers=True))

            assert frames[-1].reducers == {
                "placed": True,
                "shipped": True,
                "stage": "shipped",
            }

    def when_reflected():
        def it_projects_the_value_over_the_log():
            graph = EventGraph([ship], reducers=[placed, shipped, stage])
            log = graph.invoke(Placed())

            assert graph.reflect(log).state() == {
                "placed": True,
                "shipped": True,
                "stage": "shipped",
            }


class Asked(IntegrationEvent):
    pass


def _agui_graph() -> EventGraph:
    @on(Asked)
    def place(event: Asked) -> Placed:
        return Placed()

    return EventGraph(
        [place],
        reducers=[message_reducer(), placed, shipped, stage],
        checkpointer=MemorySaver(),
    )


def _input(**overrides: Any) -> RunAgentInput:
    fields: dict[str, Any] = {
        "thread_id": "agui-1",
        "run_id": "run-1",
        "state": {},
        "messages": [],
        "tools": [],
        "context": [],
        "forwarded_props": {},
    }
    return RunAgentInput(**(fields | overrides))


def _snapshots(events: list[Any]) -> list[dict[str, Any]]:
    return [e.snapshot for e in events if e.type == EventType.STATE_SNAPSHOT]


def describe_AGUIAdapter():
    def when_include_reducers_names_a_derived():
        async def it_includes_the_derived_that_include_reducers_names():
            adapter = AGUIAdapter(
                _agui_graph(),
                seed_factory=lambda inp: Asked(),
                include_reducers=["stage"],
            )

            events = [e async for e in adapter.stream(_input())]

            assert _snapshots(events)[-1] == {"stage": "placed"}

        async def it_includes_the_derived_in_the_snapshot_of_a_checkpoint():
            graph = _agui_graph()
            adapter = AGUIAdapter(
                graph, seed_factory=lambda inp: Asked(), include_reducers=["stage"]
            )
            [e async for e in adapter.stream(_input())]

            events = [e async for e in adapter.snapshot(_input())]

            assert _snapshots(events) == [{"stage": "placed"}]

    def when_the_client_state_holds_a_derived():
        async def it_drops_the_derived_from_frontend_state_mutated():
            seen: list[dict[str, Any]] = []

            @on(FrontendStateMutated)
            def keep(event: FrontendStateMutated) -> None:
                seen.append(dict(event.state))

            graph = EventGraph(
                [keep], reducers=[message_reducer(), placed, shipped, stage]
            )
            adapter = AGUIAdapter(graph, seed_factory=lambda inp: Asked())

            [e async for e in adapter.stream(_input(state={"stage": "x", "placed": 1}))]

            assert seen == [{"placed": 1}]
