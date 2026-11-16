"""``RunScoped``: a service derived per run from the node's ``RunnableConfig``."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

import pytest
from conftest import Started
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 - a factory annotation resolves at graph build
)

from langgraph_events import (
    Derived,
    EventGraph,
    EventLog,
    IntegrationEvent,
    Reducer,
    ReducerNotSetError,
    RunScoped,
    ScalarReducer,
    on,
)
from langgraph_events._handler import HandlerMeta
from langgraph_events._internal import _build_inject
from langgraph_events._namespace import NamespaceModel
from langgraph_events._services import NamedServices


class _Model:
    """Stand-in for a per-run value, such as a chat model bound to a language."""

    def __init__(self, language: str) -> None:
        self.language = language


def _model_for(config: RunnableConfig) -> _Model:
    return _Model(config["configurable"]["language"])


class _Other:
    """A type unrelated to ``_Model``."""


def _other_for(config: RunnableConfig) -> _Other:
    return _Other()


class _SubModel(_Model):
    """A subtype of ``_Model``."""


def _sub_model_for(config: RunnableConfig) -> _SubModel:
    return _SubModel(config["configurable"]["language"])


def _any_for(config: RunnableConfig) -> Any:
    return _model_for(config)


T = TypeVar("T")


class _Speaks(Protocol):
    """A static contract. Python cannot test it at run time."""

    def speak(self) -> str: ...


@runtime_checkable
class _Listens(Protocol):
    def listen(self) -> str: ...


@runtime_checkable
class _HasName(Protocol):
    name: str


class _Factory:
    """A class whose alternate constructor serves as the factory."""

    @classmethod
    def for_run(cls, config: RunnableConfig) -> Self:
        return cls()


@functools.cache
def _cached_for(config: RunnableConfig) -> _Model:
    return _model_for(config)


def _generic_for(config: RunnableConfig) -> T:  # type: ignore[type-var]
    return _model_for(config)  # type: ignore[return-value]


def describe_RunScoped():
    def describe_injection():
        def when_a_handler_names_a_run_scoped_service():
            def it_injects_the_factory_result_derived_from_the_run_config():
                observed: dict[str, object] = {}

                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    observed["model"] = model

                graph = EventGraph([handle], services={"model": RunScoped(_model_for)})
                graph.invoke(Started(), config={"configurable": {"language": "fr"}})

                model = observed["model"]
                assert isinstance(model, _Model)
                assert model.language == "fr"

        def when_one_node_call_handles_several_events():
            def it_calls_the_factory_once():
                calls: list[RunnableConfig] = []

                def counting(config: RunnableConfig) -> _Model:
                    calls.append(config)
                    return _model_for(config)

                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                graph = EventGraph([handle], services={"model": RunScoped(counting)})
                graph.invoke(
                    [Started(data="a"), Started(data="b")],
                    config={"configurable": {"language": "en"}},
                )

                assert len(calls) == 1

        def when_two_handlers_each_run_in_their_own_node_call():
            def it_calls_the_factory_once_per_node_call():
                calls: list[RunnableConfig] = []

                def counting(config: RunnableConfig) -> _Model:
                    calls.append(config)
                    return _model_for(config)

                @on(Started)
                def first(event: Started, model: _Model) -> None:
                    pass

                @on(Started)
                def second(event: Started, model: _Model) -> None:
                    pass

                graph = EventGraph(
                    [first, second], services={"model": RunScoped(counting)}
                )
                graph.invoke(Started(), config={"configurable": {"language": "en"}})

                assert len(calls) == 2

        def when_the_factory_raises():
            def it_reraises_the_same_error_noting_the_handler_and_parameter():
                def broken(config: RunnableConfig) -> _Model:
                    return _Model(config["configurable"]["missing"])

                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                graph = EventGraph([handle], services={"model": RunScoped(broken)})

                with pytest.raises(KeyError) as info:
                    graph.invoke(Started(), config={"configurable": {}})

                notes = getattr(info.value, "__notes__", [])
                assert any("handle" in n and "model" in n for n in notes), notes

        def when_the_runtime_config_is_missing():
            def it_raises_a_value_error_naming_the_factory_and_parameter():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                services = NamedServices({"model": RunScoped(_model_for)})
                bound = services.bind_factories({}, None).bind(
                    HandlerMeta.of(handle, readables={}), None
                )

                with pytest.raises(
                    ValueError,
                    match=r"RunScoped factory '_model_for' for service 'model' "
                    r"needs the run config for parameter 'config'",
                ):
                    _build_inject(
                        bound,
                        {"events": []},
                        {},
                        None,
                        model_provider=lambda: NamespaceModel([]),
                    )

    def describe_construction():
        def when_the_factory_is_not_callable():
            def it_raises_a_type_error():
                with pytest.raises(TypeError, match=r"callable"):
                    RunScoped(42)  # type: ignore[arg-type]

        def when_the_factory_is_a_coroutine_function():
            def it_raises_a_type_error():
                async def afactory(config: RunnableConfig) -> _Model:
                    return _model_for(config)

                with pytest.raises(TypeError, match=r"coroutine"):
                    RunScoped(afactory)

        def when_the_factory_is_an_object_whose_call_method_is_async():
            def it_raises_a_type_error():
                class AsyncCallable:
                    async def __call__(self, config: RunnableConfig) -> _Model:
                        return _model_for(config)

                with pytest.raises(TypeError, match=r"coroutine"):
                    RunScoped(AsyncCallable())

        def when_the_factory_has_no_return_annotation():
            def it_raises_a_type_error_naming_the_factory():
                def unannotated(config):  # type: ignore[no-untyped-def]
                    return _model_for(config)

                with pytest.raises(TypeError, match=r"return annotation.*unannotated"):
                    RunScoped(unannotated)

        def when_the_factory_is_a_partial():
            def it_reads_the_wrapped_function():
                RunScoped(functools.partial(_model_for))

        def when_the_factory_is_a_callable_object():
            def it_reads_the_return_annotation_of_call():
                class Callable:
                    def __call__(self, config: RunnableConfig) -> _Model:
                        return _model_for(config)

                RunScoped(Callable())

        def when_the_factory_is_a_lambda():
            def it_says_a_lambda_cannot_carry_a_return_annotation():
                with pytest.raises(TypeError, match=r"lambda cannot"):
                    RunScoped(lambda config: _Model("en"))

        def when_the_factory_inherits_an_unannotated_call_method():
            def it_names_the_object_class():
                class Base:
                    def __call__(self, config):  # type: ignore[no-untyped-def]
                        return _model_for(config)

                class Sub(Base):
                    pass

                with pytest.raises(TypeError, match=r"Sub"):
                    RunScoped(Sub())

        def when_placed_in_the_type_keyed_sequence_form():
            def it_raises_a_type_error_at_graph_construction():
                @on(Started)
                def handle(event: Started) -> None:
                    pass

                with pytest.raises(TypeError, match=r"RunScoped.*mapping"):
                    EventGraph([handle], services=[RunScoped(_model_for)])

    def describe_annotation_check():
        def when_the_factory_return_annotation_does_not_resolve():
            def it_raises_a_type_error_naming_the_factory_and_the_cause():
                def broken(config: RunnableConfig) -> _Model:
                    return _model_for(config)

                broken.__annotations__["return"] = "MissingModel"

                @on(Started)
                def handle(event: Started) -> None:
                    pass

                with pytest.raises(TypeError, match=r"broken.*MissingModel"):
                    EventGraph([handle], services={"model": RunScoped(broken)})

        def when_the_factory_returns_another_type():
            def it_raises_a_type_error_naming_handler_parameter_and_both_types():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                with pytest.raises(TypeError) as info:
                    EventGraph([handle], services={"model": RunScoped(_other_for)})

                message = str(info.value)
                assert "'handle'" in message
                assert "'model'" in message
                assert "_Model" in message
                assert "_Other" in message
                assert "RunScoped(_other_for)" in message

        def when_the_factory_returns_a_subtype():
            def it_builds_and_injects():
                seen: list[_Model] = []

                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    seen.append(model)

                graph = EventGraph(
                    [handle], services={"model": RunScoped(_sub_model_for)}
                )
                graph.invoke(Started(), config={"configurable": {"language": "fr"}})

                assert isinstance(seen[0], _SubModel)

        def when_the_factory_returns_any():
            def it_skips_the_check():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                EventGraph([handle], services={"model": RunScoped(_any_for)})

        def when_the_factory_returns_a_type_variable():
            def it_skips_the_check():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                EventGraph([handle], services={"model": RunScoped(_generic_for)})

        def when_the_parameter_annotation_is_a_protocol_that_is_not_runtime_checkable():
            def it_warns_and_builds():
                @on(Started)
                def handle(event: Started, model: _Speaks) -> None:
                    pass

                with pytest.warns(UserWarning, match=r"'model'.*not checked"):
                    EventGraph([handle], services={"model": RunScoped(_model_for)})

        def when_the_parameter_is_a_runtime_protocol_that_has_a_data_member():
            def it_warns_and_builds_for_a_run_scoped_factory():
                @on(Started)
                def handle(event: Started, model: _HasName) -> None:
                    pass

                with pytest.warns(UserWarning, match=r"'model'.*not checked"):
                    EventGraph([handle], services={"model": RunScoped(_model_for)})

        def when_the_parameter_annotation_is_a_runtime_checkable_protocol():
            def it_checks_the_factory_return_type():
                @on(Started)
                def handle(event: Started, model: _Listens) -> None:
                    pass

                with pytest.raises(TypeError, match=r"'model'.*_Listens"):
                    EventGraph([handle], services={"model": RunScoped(_model_for)})

        def when_the_same_handler_is_built_in_two_graphs():
            def it_checks_each_graph_against_its_own_services():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                EventGraph([handle], services={"model": RunScoped(_model_for)})
                with pytest.raises(TypeError, match=r"_Other"):
                    EventGraph([handle], services={"model": RunScoped(_other_for)})
                EventGraph([handle], services={"model": RunScoped(_model_for)})

        def when_the_factory_is_a_class():
            def it_uses_the_class_as_the_provided_type():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                with pytest.raises(TypeError, match=r"'model'.*_Other"):
                    EventGraph([handle], services={"model": RunScoped(_Other)})

        def when_the_annotation_nests_a_protocol_that_is_not_runtime_checkable():
            def it_warns_and_builds():
                @on(Started)
                def handle(event: Started, model: _Speaks | None) -> None:
                    pass

                with pytest.warns(UserWarning, match=r"'model'.*not checked"):
                    EventGraph([handle], services={"model": RunScoped(_model_for)})

        def when_the_factory_is_wrapped_by_a_decorator_from_another_module():
            def it_resolves_the_return_annotation_against_the_factory_module():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                EventGraph([handle], services={"model": RunScoped(_cached_for)})

        def when_the_factory_is_a_classmethod_returning_self():
            def it_compares_the_owner_class():
                @on(Started)
                def handle(event: Started, model: _Other) -> None:
                    pass

                with pytest.raises(TypeError, match=r"returns _Factory\."):
                    EventGraph(
                        [handle], services={"model": RunScoped(_Factory.for_run)}
                    )


# ---------------------------------------------------------------------------
# A factory binds its parameters like a handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Facts:
    """A run context schema."""

    language: str


class _Picked(IntegrationEvent):
    choice: str = ""


class _Told(IntegrationEvent):
    text: str = ""


_choice = ScalarReducer(name="choice", event_type=_Picked, fn=lambda e: e.choice)
_choices = Reducer(name="choices", event_type=_Picked, fn=lambda e: [e.choice])


def _shout_of(choice: str | None) -> str:
    return (choice or "none").upper()


_shout = Derived(name="shout", fn=_shout_of)


class _Reader:
    """A per-run value built from the inputs that a factory reads."""

    def __init__(self, **inputs: object) -> None:
        self.inputs = inputs


def _choice_reader(choice: str | None) -> _Reader:
    return _Reader(choice=choice)


def _shout_reader(shout: str) -> _Reader:
    return _Reader(shout=shout)


def _log_reader(log: EventLog) -> _Reader:
    return _Reader(log=log)


def _base_reader(choices: list[str]) -> _Reader:
    return _Reader(choices=choices)


def _left_of(base: _Reader) -> _Model:
    return _Model(f"left:{id(base)}")


def _right_of(base: _Reader) -> _Other:
    other = _Other()
    other.base = base  # type: ignore[attr-defined]
    return other


def _strict_reader(choice: str) -> _Reader:
    return _Reader(choice=choice)


def _setting_reader(setting: str) -> _Reader:
    return _Reader(setting=setting)


def _context_reader(facts: _Facts, run: RunnableConfig) -> _Reader:
    return _Reader(language=facts.language, thread=run["configurable"]["thread_id"])


def _defaulted_reader(choice: str = "coffee") -> _Reader:
    return _Reader(choice=choice)


def _typo_reader(choise: str) -> _Reader:
    return _Reader(choise=choise)


def _wrong_reducer_type(choices: str) -> _Reader:
    return _Reader(choices=choices)


def _wrong_derived_type(shout: int) -> _Reader:
    return _Reader(shout=shout)


def _wrong_factory_type(base: _Model) -> _Reader:
    return _Reader(base=base)


def _wrong_service_type(setting: int) -> _Reader:
    return _Reader(setting=setting)


def _wrong_log_type(log: str) -> _Reader:
    return _Reader(log=log)


def _first_of(second: _Model) -> _Reader:
    return _Reader(second=second)


def _second_of(first: _Reader) -> _Model:
    return _Model("en")


def _variadic_reader(**inputs: object) -> _Reader:
    return _Reader(**inputs)


def _raising_base(choices: list[str]) -> _Reader:
    raise LookupError("no base")


@on(_Told)
def _read_one(event: _Told, reader: _Reader) -> None:
    _SEEN.append(reader)


_SEEN: list[object] = []


@pytest.fixture(autouse=True)
def _clear_seen() -> None:
    _SEEN.clear()


def describe_RunScoped_factory_inputs():
    def when_the_factory_reads_a_stored_reducer():
        def it_receives_the_value_at_node_start():
            graph = EventGraph(
                [_read_one],
                reducers=[_choice],
                services={"reader": RunScoped(_choice_reader)},
            )

            graph.invoke([_Picked(choice="tea"), _Told()])

            assert [r.inputs for r in _SEEN] == [{"choice": "tea"}]  # type: ignore[attr-defined]

        def when_the_reducer_is_unset():
            def it_receives_the_empty_default():
                graph = EventGraph(
                    [_read_one],
                    reducers=[_choice],
                    services={"reader": RunScoped(_choice_reader)},
                )

                graph.invoke(_Told())

                assert [r.inputs for r in _SEEN] == [{"choice": None}]  # type: ignore[attr-defined]

            def when_the_annotation_rejects_none():
                def it_raises_reducer_not_set_error_naming_the_factory():
                    graph = EventGraph(
                        [_read_one],
                        reducers=[_choice],
                        services={"reader": RunScoped(_strict_reader)},
                    )

                    with pytest.raises(
                        ReducerNotSetError,
                        match=r"RunScoped factory '_strict_reader' for service "
                        r"'reader' requires reducer 'choice' to be set",
                    ):
                        graph.invoke(_Told())

    def when_a_parameter_is_positional_only():
        def it_passes_the_value_by_position():
            def positional(choice: str | None, /) -> _Reader:
                return _Reader(choice=choice)

            graph = EventGraph(
                [_read_one],
                reducers=[_choice],
                services={"reader": RunScoped(positional)},
            )

            graph.invoke([_Picked(choice="tea"), _Told()])

            assert [r.inputs for r in _SEEN] == [{"choice": "tea"}]  # type: ignore[attr-defined]

    def when_a_partial_binds_a_keyword():
        def it_binds_only_the_parameters_that_the_partial_leaves_open():
            def scaled(choice: str | None, times: int) -> _Reader:
                return _Reader(choice=choice, times=times)

            graph = EventGraph(
                [_read_one],
                reducers=[_choice],
                services={"reader": RunScoped(functools.partial(scaled, times=2))},
            )

            graph.invoke([_Picked(choice="tea"), _Told()])

            assert [r.inputs for r in _SEEN] == [{"choice": "tea", "times": 2}]  # type: ignore[attr-defined]

    def when_the_factory_reads_a_derived():
        def it_receives_the_value_that_a_handler_receives():
            @on(_Told)
            def both(event: _Told, reader: _Reader, shout: str) -> None:
                _SEEN.append((reader.inputs["shout"], shout))

            graph = EventGraph(
                [both],
                reducers=[_choice, _shout],
                services={"reader": RunScoped(_shout_reader)},
            )

            graph.invoke([_Picked(choice="tea"), _Told()])

            assert _SEEN == [("TEA", "TEA")]

    def when_the_factory_reads_log():
        def it_receives_the_log_of_the_node():
            @on(_Told)
            def both(event: _Told, reader: _Reader, log: EventLog) -> None:
                _SEEN.append((list(reader.inputs["log"]), list(log)))  # type: ignore[call-overload]

            graph = EventGraph([both], services={"reader": RunScoped(_log_reader)})

            graph.invoke([_Picked(choice="tea"), _Told()])

            ((from_factory, from_handler),) = _SEEN
            assert from_factory == from_handler
            assert len(from_factory) == 2

    def when_the_factory_reads_the_config_and_the_context():
        def it_binds_each_by_its_annotation():
            graph = EventGraph(
                [_read_one],
                context_schema=_Facts,
                services={"reader": RunScoped(_context_reader)},
            )

            graph.invoke(
                _Told(),
                config={"configurable": {"thread_id": "t1"}},
                context=_Facts("fr"),
            )

            assert [r.inputs for r in _SEEN] == [{"language": "fr", "thread": "t1"}]  # type: ignore[attr-defined]

    def when_the_factory_reads_a_plain_service():
        def it_receives_the_service_of_that_name():
            graph = EventGraph(
                [_read_one],
                services={"reader": RunScoped(_setting_reader), "setting": "on"},
            )

            graph.invoke(_Told())

            assert [r.inputs for r in _SEEN] == [{"setting": "on"}]  # type: ignore[attr-defined]

    def when_two_factories_read_one_factory():
        def it_builds_the_shared_factory_once_per_node_call():
            calls: list[list[str]] = []

            def counting_base(choices: list[str]) -> _Reader:
                calls.append(choices)
                return _Reader(choices=choices)

            @on(_Told)
            def all_three(
                event: _Told, left: _Model, right: _Other, base: _Reader
            ) -> None:
                _SEEN.append((left.language, right.base, base))  # type: ignore[attr-defined]

            graph = EventGraph(
                [all_three],
                reducers=[_choices],
                services={
                    "base": RunScoped(counting_base),
                    "left": RunScoped(_left_of),
                    "right": RunScoped(_right_of),
                },
            )

            graph.invoke(_Told())

            ((left, right_base, base),) = _SEEN
            assert len(calls) == 1
            assert right_base is base
            assert left == f"left:{id(base)}"

    def when_no_handler_declares_the_factory():
        def it_never_calls_the_factory():
            calls: list[object] = []

            def unused(choices: list[str]) -> _Reader:
                calls.append(choices)
                return _Reader()

            @on(_Told)
            def plain(event: _Told) -> None:
                pass

            graph = EventGraph(
                [plain], reducers=[_choices], services={"unused": RunScoped(unused)}
            )

            graph.invoke(_Told())

            assert calls == []

    def when_a_factory_that_another_factory_reads_raises():
        def it_notes_each_requester():
            graph = EventGraph(
                [_read_one],
                reducers=[_choices],
                services={
                    "base": RunScoped(_raising_base),
                    "reader": RunScoped(_first_of_base),
                },
            )

            with pytest.raises(LookupError) as info:
                graph.invoke(_Told())

            assert info.value.__notes__[0] == (
                "RunScoped factory '_first_of_base' for service 'reader' "
                "requested the run-scoped service 'base', but its factory raised."
            )

        def it_adds_no_note_for_the_reader_whose_factory_never_ran():
            graph = EventGraph(
                [_read_one],
                reducers=[_choices],
                services={
                    "base": RunScoped(_raising_base),
                    "reader": RunScoped(_first_of_base),
                },
            )

            with pytest.raises(LookupError) as info:
                graph.invoke(_Told())

            assert not any("service 'reader'" in n for n in info.value.__notes__[1:])


def _first_of_base(base: _Reader) -> _Reader:
    return base


def describe_RunScoped_factory_rules():
    def _build(services: dict[str, object], **kwargs: Any) -> EventGraph:
        return EventGraph(
            [_read_one],
            reducers=[_choice, _choices, _shout],
            services=services,
            **kwargs,
        )

    def when_a_parameter_has_a_default():
        def when_the_input_is_unset():
            def it_receives_the_default():
                _build({"reader": RunScoped(_defaulted_reader)}).invoke(_Told())

                assert [r.inputs for r in _SEEN] == [{"choice": "coffee"}]  # type: ignore[attr-defined]

        def when_the_input_is_set():
            def it_receives_the_value():
                _build({"reader": RunScoped(_defaulted_reader)}).invoke(
                    [_Picked(choice="tea"), _Told()]
                )

                assert [r.inputs for r in _SEEN] == [{"choice": "tea"}]  # type: ignore[attr-defined]

    def when_a_parameter_names_nothing():
        def it_raises_naming_the_other_services_and_the_fix():
            with (
                pytest.warns(UserWarning, match=r"'choise'.*Available reducers"),
                pytest.raises(TypeError) as info,
            ):
                _build({"reader": RunScoped(_typo_reader), "setting": "on"})

            assert str(info.value) == (
                "RunScoped factory '_typo_reader' for service 'reader' declares "
                "parameter(s) ['choise'] that the framework cannot inject: "
                "'choise' (str). services= has the name keys ['setting']. For "
                "service injection, register a matching instance via services=; "
                "for state, register a Reducer; otherwise remove the parameter."
            )

        def when_the_parameter_is_named_config():
            def it_names_the_run_config_fix():
                def untyped_config(config: dict[str, Any]) -> _Model:
                    return _Model("en")

                with pytest.raises(
                    TypeError,
                    match=r"To read the run config, annotate 'config' with exactly "
                    r"RunnableConfig\.",
                ):
                    _build({"reader": RunScoped(untyped_config)})

        def when_the_parameter_is_named_log():
            def it_names_the_event_log_fix():
                with pytest.raises(
                    TypeError,
                    match=r"To read the EventLog, annotate 'log' with EventLog\.",
                ):
                    _build({"reader": RunScoped(_wrong_log_type)})

        def when_the_parameter_annotation_is_an_event():
            def it_says_that_a_factory_cannot_read_the_event():
                def from_event(event: _Told) -> _Model:
                    return _Model(event.text)

                with pytest.raises(TypeError) as info:
                    _build({"reader": RunScoped(from_event)})

                assert "A factory cannot read the event _Told. Read it in the " in (
                    str(info.value)
                )
                assert "context_schema" not in str(info.value)

    def when_a_parameter_is_variadic():
        def it_raises_naming_the_factory_and_the_parameter():
            with pytest.raises(TypeError, match=r"'_variadic_reader'.*'inputs'"):
                _build({"reader": RunScoped(_variadic_reader)})

    def when_python_cannot_read_the_signature():
        def it_raises_naming_the_factory():
            with pytest.raises(TypeError, match=r"cannot read the parameters"):
                _build({"reader": RunScoped(dict)})

    def when_a_service_has_the_name_of_a_reducer():
        def it_raises_for_the_graph():
            with pytest.raises(TypeError) as info:
                _build({"reader": RunScoped(_choice_reader), "choice": "tea"})

            assert str(info.value) == (
                "services= has the name key 'choice', and a reducer has the "
                "same name. A parameter cannot name both. Rename the service "
                "or the reducer."
            )

    def when_a_service_has_the_name_of_a_derived():
        def it_raises_for_the_graph():
            with pytest.raises(TypeError, match=r"'shout', and a Derived has"):
                _build({"reader": RunScoped(_choice_reader), "shout": "x"})

    def when_a_factory_reads_log_and_a_service_has_the_name_log():
        def it_reads_the_event_log_by_the_annotation():
            graph = _build({"reader": RunScoped(_log_reader), "log": "x"})

            graph.invoke(_Told())

            assert [type(r.inputs["log"]) for r in _SEEN] == [EventLog]  # type: ignore[attr-defined]

    def when_a_reducer_has_the_name_log():
        def it_reads_the_reducer_for_a_parameter_that_names_it():
            def by_name(log: str | None) -> _Reader:
                return _Reader(log=log)

            named_log = ScalarReducer(name="log", event_type=_Picked, fn=lambda e: "x")
            graph = EventGraph(
                [_read_one],
                reducers=[named_log],
                services={"reader": RunScoped(by_name)},
            )

            graph.invoke([_Picked(), _Told()])

            assert [r.inputs for r in _SEEN] == [{"log": "x"}]  # type: ignore[attr-defined]

    def when_the_annotation_does_not_match_the_source():
        @pytest.mark.parametrize(
            ("factory", "services", "fragment"),
            [
                (_wrong_reducer_type, {}, "reducer 'choices' gives list"),
                (_wrong_derived_type, {}, "Derived 'shout' gives str"),
                (
                    _wrong_factory_type,
                    {"base": RunScoped(_choice_reader)},
                    "services['base'] is RunScoped(_choice_reader) and that "
                    "factory returns _Reader",
                ),
                (
                    _wrong_service_type,
                    {"setting": "on"},
                    "services['setting'] is an instance of str",
                ),
            ],
            ids=["reducer", "derived", "factory", "service"],
        )
        def it_raises_naming_the_factory_the_parameter_and_the_fix(
            factory: Any, services: dict[str, object], fragment: str
        ):
            with pytest.raises(TypeError) as info:
                _build({"reader": RunScoped(factory), **services})

            message = str(info.value)
            assert message.startswith(
                f"RunScoped factory {factory.__name__!r} for service 'reader' "
                f"declares parameter "
            )
            assert fragment in message
            assert " the annotation" in message

    def when_factories_read_each_other_in_a_cycle():
        def it_raises_naming_the_cycle():
            with pytest.raises(TypeError) as info:
                _build(
                    {
                        "reader": RunScoped(_choice_reader),
                        "first": RunScoped(_first_of),
                        "second": RunScoped(_second_of),
                    }
                )

            assert "RunScoped factories read each other in a cycle: " in str(info.value)
            assert "first -> second -> first" in str(info.value)


def describe_RunScoped_in_the_structure_views():
    def _graph() -> EventGraph:
        return EventGraph(
            [_read_one],
            reducers=[_choices],
            services={
                "base": RunScoped(_base_reader),
                "reader": RunScoped(_first_of_base),
            },
        )

    def when_the_factories_read_reducers_and_factories():
        def it_lists_each_factory_and_its_inputs_in_the_text():
            text = _graph().namespaces().text()

            assert "RunScoped:\n  base  (reads choices)\n  reader  (reads base)" in text

        def it_lists_each_factory_and_its_inputs_in_the_json():
            payload = _graph().namespaces().to_dict()

            assert payload["run_scoped"] == [
                {"name": "base", "inputs": [{"name": "choices", "kind": "reducer"}]},
                {"name": "reader", "inputs": [{"name": "base", "kind": "run_scoped"}]},
            ]

        def it_draws_a_reads_arrow_from_each_input_to_the_factory():
            mermaid = _graph().namespaces().mermaid()

            assert "_service_reader[[reader]]:::service" in mermaid
            assert "_reducer_choices -.->|reads| _service_base" in mermaid
            assert "_service_base -.->|reads| _service_reader" in mermaid

    def when_a_factory_reads_a_framework_input():
        def _context_graph() -> EventGraph:
            return EventGraph(
                [_read_one],
                context_schema=_Facts,
                services={"reader": RunScoped(_context_reader)},
            )

        def it_lists_the_kind_of_the_input_in_the_text():
            text = _context_graph().namespaces().text()

            assert "RunScoped:\n  reader  (reads context, config)" in text

        def it_labels_the_input_in_the_mermaid():
            mermaid = _context_graph().namespaces().mermaid()

            assert '_service_reader[["reader<br>reads: context, config"]]' in mermaid


def describe_handler_parameters_of_one_framework_type():
    def when_two_parameters_are_annotated_event_log():
        def it_gives_the_log_to_each():
            @on(_Told)
            def twice(event: _Told, first: EventLog, second: EventLog) -> None:
                _SEEN.append(first is second)

            EventGraph([twice]).invoke(_Told())

            assert _SEEN == [True]


def describe_factory_parameter_defaults():
    def when_the_parameter_matches_nothing():
        def it_says_how_to_bind_a_constant():
            def with_retries(config: RunnableConfig, retries: int = 3) -> _Model:
                return _Model("en")

            with pytest.raises(
                TypeError,
                match=r"A default does not make 'retries' optional\. Bind a constant "
                r"in a closure or a functools\.partial\.",
            ):
                EventGraph([_read_one], services={"reader": RunScoped(with_retries)})
