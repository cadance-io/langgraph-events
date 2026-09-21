"""Tests for reducer ``fn`` service parameters: `_service_params`, `_bind`,
`_BoundFn`, and the unbound guard on `collect` / `seed`.
"""

from __future__ import annotations

import operator

import pytest
from conftest import MessageReceived

from langgraph_events import BaseReducer, Reducer, ScalarReducer
from langgraph_events._reducer import _BoundFn, _service_params


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
