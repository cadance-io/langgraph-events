"""Derived reducers: a value computed from other reducers, never stored."""

import functools
from typing import Any

import pytest
from conftest import services_from
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    Derived,
    DomainEvent,
    EventGraph,
    EventLog,
    FoldReducer,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Namespace,
    Reducer,
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


placed = ScalarReducer[Placed, bool](
    name="placed", event_type=Placed, fn=lambda e: True
)
shipped = ScalarReducer[Shipped, bool](
    name="shipped", event_type=Shipped, fn=lambda e: True
)


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


def _status_of(sent: bool | None) -> str:
    return "sent" if sent else "held"


def _explicit_status(sent: bool | None) -> str:
    return "explicit"


def _flag_of(placed: str) -> str:
    return placed


def _constant() -> int:
    return 42


class Parcel(Namespace):
    sent = ScalarReducer[DomainEvent, bool](event_type=DomainEvent, fn=lambda e: True)
    status = Derived(fn=_status_of)

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
            def it_raises_at_graph_construction():
                with pytest.raises(
                    TypeError,
                    match=r"field 'stage', and a Derived has the same name",
                ):
                    EventGraph(
                        [check],
                        reducers=[placed, shipped, stage],
                        services=services_from({"stage": "from-service"}),
                    )

        def when_the_annotation_rejects_none():
            def it_raises_reducer_not_set_error():
                @on(Checked)
                def strict(event: Checked, stage: str) -> None:
                    return None

                graph = EventGraph([strict], reducers=[placed, shipped, stage])

                with pytest.raises(
                    ReducerNotSetError,
                    match=r"requires Derived 'stage' to be set.*Make Derived "
                    r"'stage' return a non-None value, or give the parameter a "
                    r"default, or change the annotation",
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
            flag = Derived(name="flag", fn=_flag_of)
            low = ScalarReducer[Placed, str](
                name="placed", event_type=Placed, fn=lambda e: "set", default="low"
            )
            high = ScalarReducer[Placed, str](
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
                clash = ScalarReducer[Placed, Any](
                    name="status", event_type=Placed, fn=str
                )

                with pytest.raises(TypeError, match=r"'status'.*reducer and a Derived"):
                    EventGraph([Parcel.Look], reducers=[clash])

            def it_keeps_an_explicit_derived_over_a_namespace_derived():
                explicit = Derived(name="status", fn=_explicit_status)
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
                    {
                        "name": "status",
                        "inputs": ["sent"],
                        "reads_log": False,
                        "namespace": "Parcel",
                    }
                ]

            def it_draws_a_reads_arrow_from_each_input_to_the_derived():
                mermaid = EventGraph([Parcel.Look]).namespaces().mermaid()

                assert "_reducer_status{{status}}:::reducer" in mermaid
                assert "_reducer_sent -.->|reads| _reducer_status" in mermaid

    def when_the_graph_is_built():
        def it_rejects_an_input_that_names_no_reducer():
            def _placd(placd: bool | None) -> bool | None:
                return placd

            typo = Derived(name="stage", fn=_placd)

            with pytest.raises(TypeError, match=r"'placd'.*\['placed', 'shipped'\]"):
                EventGraph([report], reducers=[placed, shipped, typo])

        def it_lists_the_available_names_for_an_unknown_input():
            def _stag(stag: str | None) -> str | None:
                return stag

            typo = Derived(name="typo", fn=_stag)

            with pytest.raises(
                TypeError,
                match=r"Derived 'typo' reads 'stag', which names no reducer and no "
                r"Derived\. Available reducers: \['placed', 'shipped'\]\. "
                r"Available Derived values: \['stage', 'typo'\]\. To read the "
                r"EventLog, annotate the parameter with EventLog\.",
            ):
                EventGraph([report], reducers=[placed, shipped, stage, typo])

        def when_no_reducer_has_the_name_log():
            def it_refuses_a_lambda_that_reads_log():
                sized = Derived(name="sized", fn=lambda log: log)

                with pytest.raises(
                    TypeError,
                    match=r"Derived 'sized' has no annotation on parameter 'log'\.",
                ):
                    EventGraph([report], reducers=[placed, shipped, sized])

        def it_rejects_two_derived_values_that_read_each_other():
            def _first(second: int) -> int:
                return second

            def _second(first: int) -> int:
                return first

            first = Derived(name="first", fn=_first)
            second = Derived(name="second", fn=_second)

            with pytest.raises(
                TypeError,
                match=r"Derived values read each other in a cycle: "
                r"first -> second -> first\.",
            ):
                EventGraph([report], reducers=[placed, shipped, first, second])

        def it_rejects_a_derived_that_reads_itself():
            def _itself(itself: int) -> int:
                return itself

            itself = Derived(name="itself", fn=_itself)

            with pytest.raises(
                TypeError,
                match=r"Derived values read each other in a cycle: itself -> itself\.",
            ):
                EventGraph([report], reducers=[placed, shipped, itself])

        def when_a_reducer_has_the_name_log():
            def it_reads_the_reducer_for_a_parameter_named_log():
                def _echo(log: str) -> str:
                    return log

                log_reducer = ScalarReducer[Placed, str](
                    name="log", event_type=Placed, fn=lambda e: "set"
                )
                echo = Derived(name="echo", fn=_echo)

                @on(Checked)
                def tell_echo(event: Checked, echo: str) -> Reported:
                    return Reported(stage=echo)

                graph = EventGraph([tell_echo], reducers=[log_reducer, echo])

                log = graph.invoke([Placed(), Checked()])

                assert _reports(log) == ["set"]

            def it_reads_the_event_log_for_a_parameter_annotated_event_log():
                log_reducer = ScalarReducer[Placed, str](
                    name="log", event_type=Placed, fn=str
                )
                graph = EventGraph([ship, count], reducers=[log_reducer, seen])

                log = graph.invoke(Placed())

                assert [e.n for e in log.filter(Counted)] == [2]

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
            def _scaled(placed: bool | None, factor: int = 2) -> bool | None:
                return placed

            scaled = Derived(name="scaled", fn=_scaled)

            with pytest.raises(
                TypeError,
                match=r"'factor'.*A default does not make a parameter optional\. "
                r"Bind a constant in a closure\.",
            ):
                EventGraph([report], reducers=[placed, shipped, stage, scaled])

        def it_rejects_a_builtin_function():
            builtin = Derived(name="builtin", fn=max)

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
            constant = Derived(name="constant", fn=_constant)
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


ticks = FoldReducer[int](name="ticks", event_type=Ticked, default_factory=int)
limit = ScalarReducer[Placed, int](
    name="limit", event_type=Placed, fn=lambda e: 3, default=2
)


def _remaining_of(ticks: int, limit: int) -> int:
    return limit - ticks


remaining = Derived(name="remaining", fn=_remaining_of)


@on(Checked)
def poll(event: Checked, remaining: int) -> Polled:
    return Polled(remaining=remaining)


def describe_Derived_from_two_kinds_of_reducer():
    def it_combines_a_fold_and_a_scalar():
        graph = EventGraph([poll], reducers=[ticks, limit, remaining])

        log = graph.invoke([Placed(), Ticked(), Ticked(), Checked()])

        assert [e.remaining for e in log.filter(Polled)] == [1]


class Counted(IntegrationEvent):
    n: int = 0


def _count(log: EventLog) -> int:
    return len(log)


seen = Derived(name="seen", fn=_count)


def _shipped_by(log: EventLog) -> tuple[str, ...]:
    return tuple(log.cause(e).via for e in log.filter(Shipped))


shipped_by = Derived(name="shipped_by", fn=_shipped_by)


def _loud_of(stage: str | None) -> str:
    return (stage or "").upper()


def _label_of(loud: str, seen: int) -> str:
    return f"{loud}:{seen}"


loud = Derived(name="loud", fn=_loud_of)
label = Derived(name="label", fn=_label_of)


@on(Shipped)
def count(event: Shipped, seen: int) -> Counted:
    return Counted(n=seen)


@on(Checked)
def tell(event: Checked, label: str) -> Reported:
    return Reported(stage=label)


def _history_size(history: EventLog) -> int:
    return len(history)


def describe_Derived_that_reads_the_log():
    def when_a_handler_declares_it():
        def it_receives_the_value_of_the_log_at_node_start():
            graph = EventGraph([ship, count], reducers=[seen])

            log = graph.invoke(Placed())

            assert [e.n for e in log.filter(Counted)] == [2]

        def it_reads_the_log_by_annotation_under_any_parameter_name():
            sized = Derived(name="seen", fn=_history_size)
            graph = EventGraph([ship, count], reducers=[sized])

            log = graph.invoke(Placed())

            assert [e.n for e in log.filter(Counted)] == [2]

    def when_a_sync_stream_includes_it():
        def it_shows_the_value_of_the_log_in_each_frame():
            graph = EventGraph([ship, count], reducers=[seen])

            frames = list(graph.stream_events(Placed(), include_reducers=["seen"]))

            assert [f.reducers for f in frames] == [
                {"seen": 1},
                {"seen": 2},
                {"seen": 3},
            ]

    def when_an_async_stream_includes_it():
        async def it_shows_the_value_of_the_log_in_each_frame():
            graph = EventGraph([ship, count], reducers=[seen])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(), include_reducers=["seen"], include_custom_events=True
                )
            ]

            assert [f.reducers for f in frames] == [
                {"seen": 1},
                {"seen": 2},
                {"seen": 3},
            ]

        async def it_marks_the_value_changed_on_each_event():
            graph = EventGraph([ship, count], reducers=[seen])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(), include_reducers=["seen"], include_custom_events=True
                )
            ]

            assert [f.changed_reducers for f in frames] == [frozenset({"seen"})] * 3

        async def it_reads_the_causes_of_the_streamed_events():
            graph = EventGraph([ship], reducers=[shipped_by])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(),
                    include_reducers=["shipped_by"],
                    include_custom_events=True,
                )
            ]

            assert frames[-1].reducers == {"shipped_by": ("ship",)}

        async def it_counts_the_checkpointed_events_of_a_later_run():
            graph = EventGraph([ship], reducers=[seen], checkpointer=MemorySaver())
            await graph.ainvoke(Placed(), config=_config("log-1"))

            frames = [
                f
                async for f in graph.astream_events(
                    Checked(),
                    include_reducers=["seen"],
                    include_custom_events=True,
                    config=_config("log-1"),
                )
            ]

            assert [f.reducers for f in frames] == [{"seen": 3}]

    def when_reflected():
        def it_projects_the_value_over_the_log():
            graph = EventGraph([ship], reducers=[seen, shipped_by])
            log = graph.invoke(Placed())

            assert graph.reflect(log).state() == {"seen": 2, "shipped_by": ("ship",)}

    def when_the_structure_is_rendered():
        def it_lists_log_as_an_input_in_the_text():
            text = EventGraph([count], reducers=[seen]).namespaces().text()

            assert "  seen  (reads log)" in text

        def when_a_derived_has_no_parameter():
            def it_lists_no_input_in_the_text():
                constant = Derived(name="constant", fn=_constant)

                text = (
                    EventGraph([count], reducers=[seen, constant]).namespaces().text()
                )

                assert "  constant  (reads no input)" in text

        def it_states_the_log_read_in_the_json():
            derived = EventGraph([count], reducers=[seen]).namespaces().to_dict()

            assert derived["derived"] == [
                {"name": "seen", "inputs": [], "reads_log": True, "namespace": None}
            ]

        def it_labels_the_log_read_in_the_mermaid():
            mermaid = EventGraph([count], reducers=[seen]).namespaces().mermaid()

            assert '_reducer_seen{{"seen<br>reads: log"}}:::reducer' in mermaid

        def it_draws_no_reads_arrow_from_the_log():
            mermaid = EventGraph([count], reducers=[seen]).namespaces().mermaid()

            assert "-.->|reads| _reducer_seen" not in mermaid


def describe_Derived_that_reads_another_derived():
    def when_a_handler_declares_it():
        def it_receives_the_value_computed_from_the_other_derived():
            graph = EventGraph(
                [tell], reducers=[placed, shipped, stage, loud, seen, label]
            )

            log = graph.invoke([Placed(), Checked()])

            assert _reports(log) == ["PLACED:2"]

    def when_an_async_stream_includes_it():
        async def it_marks_the_value_changed_on_an_input_change_of_the_other():
            graph = EventGraph([ship], reducers=[placed, shipped, stage, loud])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(), include_reducers=["loud"], include_custom_events=True
                )
            ]

            assert [f.reducers for f in frames] == [
                {"loud": "PLACED"},
                {"loud": "SHIPPED"},
            ]
            assert [f.changed_reducers for f in frames] == [frozenset({"loud"})] * 2

        async def it_marks_nothing_changed_for_an_event_that_no_input_folds():
            graph = EventGraph([check], reducers=[placed, shipped, stage, loud])

            frames = [
                f
                async for f in graph.astream_events(
                    Checked(), include_reducers=["loud"], include_custom_events=True
                )
            ]

            assert [f.changed_reducers for f in frames] == [frozenset(), frozenset()]

    def when_a_sync_stream_includes_it():
        def it_shows_the_value_in_each_frame():
            graph = EventGraph([ship], reducers=[placed, shipped, stage, loud])

            frames = list(graph.stream_events(Placed(), include_reducers=["loud"]))

            assert [f.reducers for f in frames] == [
                {"loud": "PLACED"},
                {"loud": "SHIPPED"},
            ]

    def when_reflected():
        def it_projects_the_value_over_the_log():
            graph = EventGraph(
                [ship], reducers=[placed, shipped, stage, loud, seen, label]
            )
            log = graph.invoke(Placed())

            assert graph.reflect(log).state()["label"] == "SHIPPED:2"

    def when_the_structure_is_rendered():
        def it_lists_each_input_in_the_json():
            graph = EventGraph(
                [tell], reducers=[placed, shipped, stage, loud, seen, label]
            )

            derived = graph.namespaces().to_dict()["derived"]

            assert {
                "name": "label",
                "inputs": ["loud", "seen"],
                "reads_log": False,
                "namespace": None,
            } in derived

        def it_draws_a_reads_arrow_from_the_other_derived():
            graph = EventGraph(
                [tell], reducers=[placed, shipped, stage, loud, seen, label]
            )

            mermaid = graph.namespaces().mermaid()

            assert "_reducer_loud -.->|reads| _reducer_label" in mermaid


class _Calls:
    """Count the calls of the base function of a diamond of Derived values."""

    def __init__(self) -> None:
        self.base = 0

    def count_base(self, placed: bool | None) -> int:
        self.base += 1
        return 1 if placed else 0


def _left_of(base: int) -> int:
    return base + 1


def _right_of(base: int) -> int:
    return base + 2


def _top_of(left: int, right: int) -> int:
    return left + right


def _diamond(calls: _Calls) -> list[Derived[Any]]:
    return [
        Derived(name="base", fn=calls.count_base),
        Derived(name="left", fn=_left_of),
        Derived(name="right", fn=_right_of),
        Derived(name="top", fn=_top_of),
    ]


@on(Checked)
def read_top(event: Checked, top: int) -> Polled:
    return Polled(remaining=top)


def describe_Derived_computed_once_per_state_read():
    def when_a_handler_reads_a_diamond_of_derived_values():
        def it_computes_the_shared_input_once():
            calls = _Calls()
            graph = EventGraph([read_top], reducers=[placed, *_diamond(calls)])

            log = graph.invoke(Checked())

            assert [e.remaining for e in log.filter(Polled)] == [3]
            assert calls.base == 1

    def when_a_sync_stream_shows_each_value():
        def it_computes_the_shared_input_once_per_frame():
            calls = _Calls()
            graph = EventGraph([ship], reducers=[placed, *_diamond(calls)])

            frames = list(graph.stream_events(Placed(), include_reducers=True))

            assert len(frames) == 2
            assert calls.base == 2

    def when_reflected():
        def it_computes_the_shared_input_once():
            calls = _Calls()
            graph = EventGraph([ship], reducers=[placed, *_diamond(calls)])
            log = graph.invoke(Placed())

            assert graph.reflect(log).state()["top"] == 5
            assert calls.base == 1


class Seeded(IntegrationEvent):
    x: int = 0


def _seeded_items(e: Seeded) -> list[int]:
    return [e.x]


seeded = Reducer[Seeded, int](name="seeded", event_type=Seeded, fn=_seeded_items)


@on(Seeded)
def note_seed(event: Seeded) -> Reported:
    return Reported()


def _whole_log(log: EventLog) -> EventLog:
    return log


placed_view = Derived(name="placed_view", fn=_whole_log)


def _offset_count(offset: int, log: EventLog) -> int:
    return len(log) + offset


class _CallableCount:
    def __call__(self, log: EventLog) -> int:
        return len(log)


def describe_Derived_in_a_per_event_stream():
    def when_a_value_keeps_the_log():
        async def it_keeps_the_causes_of_an_earlier_frame():
            graph = EventGraph([ship], reducers=[placed_view])

            frames = [
                f
                async for f in graph.astream_events(
                    Placed(), include_reducers=True, include_custom_events=True
                )
            ]

            view = frames[0].reducers["placed_view"]
            assert [view.cause(e) for e in view] == [None]

    def when_the_run_has_several_seeds():
        async def it_shows_the_state_up_to_each_seed():
            graph = EventGraph([note_seed], reducers=[seeded])

            frames = [
                f
                async for f in graph.astream_events(
                    [Seeded(x=1), Seeded(x=2)],
                    include_reducers=True,
                    include_custom_events=True,
                )
            ]

            assert [f.reducers["seeded"] for f in frames[:2]] == [[1], [1, 2]]


def describe_Derived_from_a_callable_that_is_not_a_function():
    def when_the_function_is_a_partial():
        def it_reads_the_log_by_the_annotation_of_the_wrapped_function():
            sized = Derived(name="seen", fn=functools.partial(_offset_count, 10))
            graph = EventGraph([ship, count], reducers=[sized])

            log = graph.invoke(Placed())

            assert [e.n for e in log.filter(Counted)] == [12]

    def when_the_function_is_a_callable_object():
        def it_reads_the_log_by_the_annotation_of_call():
            sized = Derived(name="seen", fn=_CallableCount())
            graph = EventGraph([ship, count], reducers=[sized])

            log = graph.invoke(Placed())

            assert [e.n for e in log.filter(Counted)] == [2]


def _log_size(log: EventLog) -> int:
    return len(log)


def describe_Derived_from_a_partial_that_binds_the_log():
    def it_keeps_the_value_that_the_partial_binds():
        bound_log = Derived(name="seen", fn=functools.partial(_log_size, log=()))
        graph = EventGraph([ship, count], reducers=[bound_log])

        log = graph.invoke(Placed())

        assert [e.n for e in log.filter(Counted)] == [0]


def _defaulted_log_size(log: EventLog = ()) -> int:  # type: ignore[assignment]
    return len(log)


def describe_Derived_log_parameter_default():
    def it_refuses_the_default():
        defaulted = Derived(name="seen", fn=_defaulted_log_size)

        with pytest.raises(
            TypeError,
            match=r"Derived 'seen' gives parameter 'log' a default\. A default does "
            r"not make a parameter optional",
        ):
            EventGraph([ship, count], reducers=[defaulted])


def describe_Derived_return_annotation():
    def when_the_function_has_no_return_annotation():
        def it_raises_at_graph_build_naming_the_fix():
            def _bare(placed: bool | None):  # type: ignore[no-untyped-def]
                return placed

            unannotated = Derived(name="unannotated", fn=_bare)

            with pytest.raises(
                TypeError,
                match=r"Derived 'unannotated' has no return annotation\. .*A "
                r"lambda cannot have one, so use a def",
            ):
                EventGraph([report], reducers=[placed, shipped, stage, unannotated])

    def when_a_handler_annotation_does_not_fit_the_return_annotation():
        def it_raises_at_graph_build():
            @on(Checked)
            def wrong(event: Checked, stage: int) -> None:
                return None

            with pytest.raises(
                TypeError,
                match=r"declares parameter 'stage' as int, but Derived 'stage' "
                r"gives str \| None",
            ):
                EventGraph([wrong], reducers=[placed, shipped, stage])
