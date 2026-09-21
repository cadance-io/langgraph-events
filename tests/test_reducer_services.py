"""Tests for reducer ``fn`` service parameters: `_service_params`, `_bind`,
`_BoundFn`, the unbound guard on `collect` / `seed`, and binding a whole
reducer dict to a run config via `bind_reducers`.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import pytest
from conftest import MessageReceived

from langgraph_events import BaseReducer, Reducer, RunScoped, ScalarReducer
from langgraph_events._internal import _caller_config, bind_reducers
from langgraph_events._reducer import _BoundFn, _service_params

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _project_list(event, language):
    return [f"{language}:{event.text}"]


def _project_scalar(event, language):
    return f"{language}:{event.text}"


def _make_reducer(reducer_cls, **kwargs):
    fn = _project_list if reducer_cls is Reducer else _project_scalar
    params = {
        "name": "notes",
        "event_type": MessageReceived,
        "fn": fn,
        "namespace": None,
    }
    params.update(kwargs)
    return reducer_cls(**params)


class _Marker:
    """Sentinel ``namespace`` value. No event ever matches it."""


def _collected_value(reducer_cls, result):
    return result[0] if reducer_cls is Reducer else result


def _language_for(config: RunnableConfig) -> str:
    return config["configurable"]["language"]


def describe_service_params():
    def it_returns_empty_for_a_one_parameter_fn():
        def fn(event):
            return None

        assert _service_params(fn) == ()

    def it_returns_the_names_of_extra_parameters_in_order():
        def fn(event, language):
            return None

        assert _service_params(fn) == ("language",)

    def it_skips_a_parameter_that_has_a_default():
        def fn(event, language="en"):
            return None

        assert _service_params(fn) == ()

    def it_returns_empty_for_an_unreadable_signature():
        assert _service_params(str) == ()

    def it_returns_empty_for_attrgetter():
        assert _service_params(operator.attrgetter("text")) == ()

    def it_raises_for_a_required_positional_only_parameter():
        def fn(event, language, /):
            return None

        with pytest.raises(TypeError, match="language"):
            _service_params(fn)

    def it_treats_a_keyword_only_parameter_as_a_service_parameter():
        def fn(event, *, language):
            return None

        assert _service_params(fn) == ("language",)

    def it_returns_empty_for_a_bound_fn():
        def fn(event, language):
            return None

        bound = _BoundFn(fn, {"language": "fr"})
        assert _service_params(bound) == ()


def describe_bind():
    def when_the_fn_has_no_service_parameter():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_returns_the_same_object(reducer_cls):
            fn = (lambda event: []) if reducer_cls is Reducer else (lambda event: None)
            reducer = reducer_cls(name="notes", event_type=MessageReceived, fn=fn)

            assert reducer._bind({"language": "fr"}) is reducer

    def when_the_fn_has_a_service_parameter():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_returns_a_different_object(reducer_cls):
            reducer = _make_reducer(reducer_cls)

            bound = reducer._bind({"language": "fr"})

            assert bound is not reducer

        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_collects_using_the_bound_value(reducer_cls):
            reducer = _make_reducer(reducer_cls)
            bound = reducer._bind({"language": "fr"})

            result = bound.collect([MessageReceived(text="hi")])

            assert _collected_value(reducer_cls, result) == "fr:hi"

        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_keeps_the_original_name_and_namespace(reducer_cls):
            # A namespace unrelated to MessageReceived on purpose: this test
            # does not call collect, so the namespace filter never runs.
            reducer = _make_reducer(reducer_cls, namespace=_Marker)

            bound = reducer._bind({"language": "fr"})

            assert bound.name == reducer.name
            assert bound.namespace is _Marker

        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_does_not_show_the_service_value_in_repr(reducer_cls):
            reducer = _make_reducer(reducer_cls)

            bound = reducer._bind({"language": "sk-secret"})

            assert "sk-secret" not in repr(bound)

        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_ignores_an_extra_key_in_values(reducer_cls):
            reducer = _make_reducer(reducer_cls)

            bound = reducer._bind({"language": "fr", "unused": "x"})

            result = bound.collect([MessageReceived(text="hi")])
            assert _collected_value(reducer_cls, result) == "fr:hi"


def describe_unbound_guard():
    def when_collect_is_called_on_an_unbound_reducer():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_raises_naming_the_reducer_and_the_service_parameter(reducer_cls):
            reducer = _make_reducer(reducer_cls)

            with pytest.raises(TypeError, match="notes") as info:
                reducer.collect([MessageReceived(text="hi")])

            assert "language" in str(info.value)

    def when_seed_is_called_on_an_unbound_reducer():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_raises_naming_the_reducer_and_the_service_parameter(reducer_cls):
            reducer = _make_reducer(reducer_cls)

            with pytest.raises(TypeError, match="notes") as info:
                reducer.seed([MessageReceived(text="hi")])

            assert "language" in str(info.value)


def describe_base_reducer_defaults():
    class _NotADataclass(BaseReducer):
        """A concrete `BaseReducer` subclass that is not a dataclass."""

        name = "not_a_dataclass"
        event_type = MessageReceived
        namespace = None

        def state_annotation(self):
            return list

        @property
        def empty(self):
            return []

        def collect(self, events):
            return []

        def has_contributions(self, result):
            return bool(result)

        def output_type(self):
            return list

        def seed(self, events):
            return []

    def it_has_no_service_parameters():
        reducer = _NotADataclass()

        assert reducer._service_params == ()

    def it_binds_to_itself():
        reducer = _NotADataclass()

        assert reducer._bind({"language": "fr"}) is reducer


def describe_caller_config():
    def when_config_is_none():
        def it_returns_an_empty_configurable():
            assert _caller_config(None) == {"configurable": {}}

    def when_config_holds_internal_and_checkpoint_keys():
        def it_drops_them():
            config: RunnableConfig = {
                "configurable": {
                    "__pregel_runtime": object(),
                    "__lge_deadline": 1.0,
                    "checkpoint_ns": "ns",
                    "checkpoint_id": "id",
                    "thread_id": "t1",
                    "language": "fr",
                }
            }

            result = _caller_config(config)

            assert result == {"configurable": {"thread_id": "t1", "language": "fr"}}

    def when_config_holds_top_level_metadata():
        def it_omits_metadata_callbacks_and_tags():
            config: RunnableConfig = {
                "configurable": {"thread_id": "t1"},
                "metadata": {"a": 1},
                "callbacks": [object()],
                "tags": ["x"],
            }

            result = _caller_config(config)

            assert result == {"configurable": {"thread_id": "t1"}}
            assert "metadata" not in result
            assert "callbacks" not in result
            assert "tags" not in result


def describe_bind_reducers():
    def when_no_reducer_declares_a_service_parameter():
        def it_returns_the_same_dict_object():
            reducer = Reducer(name="notes", event_type=MessageReceived, fn=lambda e: [])
            reducers = {"notes": reducer}

            result = bind_reducers(reducers, None, None)

            assert result is reducers

    def when_a_run_scoped_service_is_declared():
        def it_resolves_the_service_from_configurable():
            reducer = _make_reducer(Reducer)
            config: RunnableConfig = {"configurable": {"language": "fr"}}

            bound = bind_reducers(
                {"notes": reducer},
                {"language": RunScoped(_language_for)},
                config,
            )

            result = bound["notes"].collect([MessageReceived(text="hi")])
            assert result == ["fr:hi"]

    def when_a_plain_service_is_declared():
        def it_binds_the_plain_value():
            reducer = _make_reducer(Reducer)

            bound = bind_reducers({"notes": reducer}, {"language": "fr"}, None)

            result = bound["notes"].collect([MessageReceived(text="hi")])
            assert result == ["fr:hi"]

    def when_two_reducers_share_one_run_scoped_service():
        def it_calls_the_factory_one_time():
            calls: list[Any] = []

            def factory(config: RunnableConfig) -> str:
                calls.append(config)
                return "fr"

            first = _make_reducer(Reducer)
            second = _make_reducer(ScalarReducer)
            config: RunnableConfig = {"configurable": {}}

            bind_reducers(
                {"first": first, "second": second},
                {"language": RunScoped(factory)},
                config,
            )

            assert len(calls) == 1

        def it_gives_the_factory_exactly_the_normalised_config():
            calls: list[Any] = []

            def factory(config: RunnableConfig) -> str:
                calls.append(config)
                return "fr"

            reducer = _make_reducer(Reducer)
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "t1",
                    "__pregel_runtime": object(),
                    "checkpoint_ns": "ns",
                },
                "metadata": {"a": 1},
            }

            bind_reducers({"notes": reducer}, {"language": RunScoped(factory)}, config)

            assert calls == [{"configurable": {"thread_id": "t1"}}]

    def when_the_factory_raises():
        def it_keeps_the_exception_type_and_carries_the_note():
            def broken(config: RunnableConfig) -> str:
                raise ValueError("boom")

            reducer = _make_reducer(Reducer)

            with pytest.raises(ValueError, match="boom") as info:
                bind_reducers(
                    {"notes": reducer},
                    {"language": RunScoped(broken)},
                    {"configurable": {}},
                )

            notes = getattr(info.value, "__notes__", [])
            assert any("notes" in n and "language" in n for n in notes), notes
