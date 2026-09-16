"""``RunScoped``: a service derived per run from the node's ``RunnableConfig``."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

        def when_placed_in_the_type_keyed_sequence_form():
            def it_raises_a_type_error_at_graph_construction():
                @on(Started)
                def handle(event: Started) -> None:
                    pass

                with pytest.raises(TypeError, match=r"RunScoped.*mapping"):
                    EventGraph([handle], services=[RunScoped(_model_for)])
