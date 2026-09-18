"""``RunScoped``: a service derived per run from the node's ``RunnableConfig``."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar, runtime_checkable

import pytest
from conftest import Started

from langgraph_events import EventGraph, RunScoped, on
from langgraph_events._handler import extract_handler_meta
from langgraph_events._internal import _build_inject
from langgraph_events._namespace import NamespaceModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


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
            def it_raises_a_value_error_naming_the_handler_and_parameter():
                @on(Started)
                def handle(event: Started, model: _Model) -> None:
                    pass

                meta = extract_handler_meta(handle, service_names=frozenset({"model"}))

                with pytest.raises(ValueError, match=r"handle.*model"):
                    _build_inject(
                        meta,
                        {"events": []},
                        {},
                        None,
                        services_by_name={"model": RunScoped(_model_for)},
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
                def handle(event: Started, model: _Other) -> None:
                    pass

                with pytest.raises(TypeError, match=r"'model'.*_Model"):
                    EventGraph([handle], services={"model": RunScoped(_Model)})

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
