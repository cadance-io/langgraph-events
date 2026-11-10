"""Derived reducers: a value computed from other reducers, never stored."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    Derived,
    DomainEvent,
    EventGraph,
    FoldReducer,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Namespace,
    ReducerNotSetError,
    Resumed,
    ScalarReducer,
    on,
)
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


class StageAsked(Interrupted):
    pass


class Answered(IntegrationEvent):
    pass


@on(Placed)
def ask(event: Placed) -> StageAsked:
    return StageAsked()


@on(Answered)
def answered(event: Answered, stage: str | None) -> Reported:
    return Reported(stage=stage or "unset")


@on(Checked)
def read_flag(event: Checked, flag: str) -> Reported:
    return Reported(stage=flag)


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

        def when_a_service_has_the_same_name():
            def it_injects_the_derived_value():
                graph = EventGraph(
                    [check],
                    reducers=[placed, shipped, stage],
                    services={"stage": "from-service"},
                )

                log = graph.invoke([Placed(), Checked()])

                assert _reports(log) == ["placed"]

        def when_the_annotation_rejects_none():
            def it_raises_reducer_not_set_error():
                @on(Checked)
                def strict(event: Checked, stage: str) -> None:
                    return None

                graph = EventGraph([strict], reducers=[placed, shipped, stage])

                with pytest.raises(
                    ReducerNotSetError,
                    match=r"requires Derived 'stage' to be set.*Make Derived "
                    r"'stage' return a non-None value, or change the annotation",
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

    def when_one_derived_serves_two_graphs():
        def it_reads_the_empty_defaults_of_each_graph():
            flag = Derived(name="flag", fn=lambda placed: placed)
            low = ScalarReducer(
                name="placed", event_type=Placed, fn=lambda e: "set", default="low"
            )
            high = ScalarReducer(
                name="placed", event_type=Placed, fn=lambda e: "set", default="high"
            )
            first = EventGraph([read_flag], reducers=[low, flag])
            second = EventGraph([read_flag], reducers=[high, flag])

            # reflect() projects from the log, so an unset input reads the
            # empty default of the reducer that this graph registers.
            assert first.reflect(first.invoke(Checked())).state()["flag"] == "low"
            assert second.reflect(second.invoke(Checked())).state()["flag"] == "high"

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

        async def it_refuses_to_apre_seed_the_value():
            graph = EventGraph(
                [report], reducers=[placed, shipped, stage], checkpointer=MemorySaver()
            )

            with pytest.raises(ValueError, match=r"Derived name\(s\) \['stage'\]"):
                await graph.apre_seed(_config("t7"), {"stage": "shipped"})

        def it_shows_the_value_in_a_sync_resume_stream():
            graph = EventGraph(
                [ask, answered],
                reducers=[placed, shipped, stage],
                checkpointer=MemorySaver(),
            )
            graph.invoke(Placed(), config=_config("t6"))

            frames = list(
                graph.stream_resume(
                    Answered(), include_reducers=["stage"], config=_config("t6")
                )
            )

            assert [type(f.event) for f in frames] == [
                Placed,
                StageAsked,
                Answered,
                Resumed,
                Reported,
            ]
            assert [f.reducers for f in frames] == [{"stage": "placed"}] * 5

    def when_declared_on_a_namespace():
        def it_takes_the_attribute_name():
            graph = EventGraph([Parcel.Look])

            log = graph.invoke(Parcel.Look())

            assert [e.status for e in log.filter(Parcel.Look.Looked)] == ["held"]

        def it_takes_the_enclosing_namespace():
            assert Parcel.status.namespace is Parcel

        def it_is_injectable_but_not_a_stored_reducer():
            topology = EventGraph([Parcel.Look]).topology

            assert topology.reducer_names == frozenset({"sent"})
            assert topology.injectable_names == frozenset({"sent", "status"})

        def when_an_explicit_entry_has_the_same_name():
            def it_rejects_an_explicit_derived_named_like_a_namespace_reducer():
                clash = Derived(name="sent", fn=lambda: True)

                with pytest.raises(TypeError, match=r"'sent'.*reducer and a Derived"):
                    EventGraph([Parcel.Look], reducers=[clash])

            def it_rejects_an_explicit_reducer_named_like_a_namespace_derived():
                clash = ScalarReducer(name="status", event_type=Placed, fn=str)

                with pytest.raises(TypeError, match=r"'status'.*reducer and a Derived"):
                    EventGraph([Parcel.Look], reducers=[clash])

            def it_keeps_an_explicit_derived_over_a_namespace_derived():
                explicit = Derived(name="status", fn=lambda sent: "explicit")
                graph = EventGraph([Parcel.Look], reducers=[explicit])

                log = graph.invoke(Parcel.Look())

                assert [e.status for e in log.filter(Parcel.Look.Looked)] == [
                    "explicit"
                ]

        def when_the_structure_is_rendered():
            def it_lists_each_input_of_the_derived_in_the_text():
                text = EventGraph([Parcel.Look]).namespaces().text()

                assert "Derived:\n  status  (reads sent)" in text

            def it_lists_each_input_of_the_derived_in_the_json():
                payload = EventGraph([Parcel.Look]).namespaces().to_dict()

                assert payload["derived"] == [
                    {"name": "status", "inputs": ["sent"], "namespace": "Parcel"}
                ]

            def it_draws_a_reads_arrow_from_each_input_to_the_derived():
                mermaid = EventGraph([Parcel.Look]).namespaces().mermaid()

                assert "_reducer_status{{status}}:::reducer" in mermaid
                assert "_reducer_sent -.->|reads| _reducer_status" in mermaid

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

            with pytest.raises(
                ValueError,
                match=r"Derived name\(s\) \['events'\] conflict with reserved",
            ):
                EventGraph([report], reducers=[placed, shipped, reserved])

        def it_rejects_an_empty_name():
            nameless = Derived(fn=lambda placed: placed)

            with pytest.raises(ValueError, match=r"Derived .* has no name"):
                EventGraph([report], reducers=[placed, shipped, nameless])

        def it_rejects_a_defaulted_parameter_that_names_no_reducer():
            scaled = Derived(name="scaled", fn=lambda placed, factor=2: placed)

            with pytest.raises(
                TypeError,
                match=r"'factor'.*A parameter with a default must still name a "
                r"reducer\. Bind a constant in a closure\.",
            ):
                EventGraph([report], reducers=[placed, shipped, stage, scaled])

        def it_rejects_a_builtin_function():
            builtin = Derived(name="builtin", fn=bool)

            with pytest.raises(
                TypeError, match=r"Derived 'builtin' cannot read the parameters"
            ):
                EventGraph([report], reducers=[placed, shipped, stage, builtin])

        def it_rejects_a_function_that_is_not_callable():
            broken = Derived(name="broken", fn="placed")  # type: ignore[arg-type]

            with pytest.raises(
                TypeError, match=r"Derived 'broken' cannot read the parameters"
            ):
                EventGraph([report], reducers=[placed, shipped, stage, broken])

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

        async def it_marks_nothing_changed_for_an_event_that_no_input_folds():
            graph = EventGraph([check], reducers=[placed, shipped, stage])

            frames = [
                f
                async for f in graph.astream_events(
                    Checked(), include_reducers=["stage"], include_custom_events=True
                )
            ]

            assert [type(f.event) for f in frames] == [Checked, Reported]
            assert [f.changed_reducers for f in frames] == [frozenset(), frozenset()]

        async def it_never_marks_a_zero_parameter_derived_changed():
            constant = Derived(name="constant", fn=lambda: 42)
            graph = EventGraph([ship], reducers=[placed, shipped, constant])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(),
                    include_reducers=["placed", "constant"],
                    include_custom_events=True,
                )
            ]

            assert [f.reducers for f in frames] == [
                {"placed": True, "constant": 42},
                {"placed": True, "constant": 42},
            ]
            assert [f.changed_reducers for f in frames] == [
                frozenset({"placed"}),
                frozenset(),
            ]

        def it_lists_every_value_in_registration_order_for_true():
            graph = EventGraph([ship], reducers=[stage, placed, shipped])

            frames = list(graph.stream_events(Placed(), include_reducers=True))

            assert list(frames[-1].reducers) == ["stage", "placed", "shipped"]

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


class Ticked(IntegrationEvent):
    def fold(self, state: int) -> int:
        return state + 1


class Polled(IntegrationEvent):
    remaining: int


ticks = FoldReducer(name="ticks", event_type=Ticked, default_factory=int)
limit = ScalarReducer(name="limit", event_type=Placed, fn=lambda e: 3, default=2)
remaining = Derived(name="remaining", fn=lambda ticks, limit: limit - ticks)


@on(Checked)
def poll(event: Checked, remaining: int) -> Polled:
    return Polled(remaining=remaining)


def describe_Derived_from_two_kinds_of_reducer():
    def it_combines_a_fold_and_a_scalar():
        graph = EventGraph([poll], reducers=[ticks, limit, remaining])

        log = graph.invoke([Placed(), Ticked(), Ticked(), Checked()])

        assert [e.remaining for e in log.filter(Polled)] == [1]
