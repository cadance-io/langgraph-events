"""Tests for reducer ``fn`` service parameters: `_service_params`, `_bind`,
`_BoundFn`, the unbound guard on `collect` / `seed`, binding a whole reducer
dict to a run config via `bind_reducers`, the graph build check,
`EventGraph._reducers_for`, the node paths, the stream shadow and the AG-UI
adapter.
"""

from __future__ import annotations

import inspect
import operator
from typing import TYPE_CHECKING, Any

import pytest
from conftest import MessageReceived
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    BaseReducer,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    EventLog,
    FoldReducer,
    IntegrationEvent,
    Namespace,
    Reducer,
    Reflection,
    RunScoped,
    ScalarReducer,
    message_reducer,
    on,
)
from langgraph_events._internal import _caller_config, bind_reducers
from langgraph_events._reducer import _BoundFn, _fold_service_params, _service_params
from langgraph_events.agui import AGUIAdapter, FrontendStateMutated

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


def _project_focus(event, language):
    return [f"{language}:{event.state['focus']}"]


# Module-level, so each stream test can read what the factory received.
_factory_configs: list[Any] = []


def _recording_language_for(config: RunnableConfig) -> str:
    """Record the config the framework passes, then return the language."""
    _factory_configs.append(config)
    return config["configurable"]["language"]


_metadata_seen: list[Any] = []


def _metadata_probe(config: RunnableConfig) -> str:
    """Record the top-level ``metadata`` key, then return a fixed language."""
    _metadata_seen.append(config.get("metadata"))
    return "fr"


@on(MessageReceived)
def _noop(event: MessageReceived) -> None:
    return None


class Pinged(IntegrationEvent):
    """A seed event that never matches the ``notes`` reducer.

    Triggers ``_relay`` below, so the reducer's only contribution comes
    from the handler-produced ``MessageReceived``, not from seeding.
    """


@on(Pinged)
def _relay(event: Pinged) -> MessageReceived:
    return MessageReceived(text="hi")


# Module-level so the Reflection annotation below resolves at runtime.
_reflection_states: list[dict] = []


@on(MessageReceived)
def _capture_reflection_state(event: MessageReceived, run: Reflection) -> None:
    _reflection_states.append(run.state())


# Module-level so the return annotation below resolves at runtime — see the
# same pattern in test_reducer_namespace.py. The inline handler lives on the
# Command itself, since only a Command's own handler may emit its outcome.
class _NamespaceWithService(Namespace):
    notes = Reducer(event_type=Event, fn=_project_list)

    class Act(Command):
        class Acted(DomainEvent):
            pass

        def handle(self) -> _NamespaceWithService.Act.Acted:
            return _NamespaceWithService.Act.Acted()


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


def describe_cached_service_params():
    """`_service_params` reads the `fn` signature at most one time per
    reducer instance. See issue #193 final fix wave, Fix 1."""

    def when_collect_runs_more_than_once():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_reads_the_fn_signature_at_most_one_time(reducer_cls, monkeypatch):
            calls: list[Any] = []
            real_signature = inspect.signature

            def counting_wrapper(obj: Any, *args: Any, **kwargs: Any) -> Any:
                calls.append(obj)
                return real_signature(obj, *args, **kwargs)

            monkeypatch.setattr(
                "langgraph_events._reducer.inspect.signature", counting_wrapper
            )
            fn = (lambda event: []) if reducer_cls is Reducer else (lambda event: None)
            reducer = reducer_cls(name="notes", event_type=MessageReceived, fn=fn)

            reducer.collect([MessageReceived(text="hi")])
            reducer.collect([MessageReceived(text="hi")])
            reducer.collect([MessageReceived(text="hi")])

            assert len(calls) <= 1

    def when_the_original_service_params_was_already_read():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_does_not_carry_the_cache_into_a_bound_copy(reducer_cls):
            reducer = _make_reducer(reducer_cls)
            assert reducer._service_params == ("language",)  # populates the cache

            bound = reducer._bind({"language": "fr"})

            assert bound._service_params == ()
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


def describe_fold_service_params():
    def it_returns_empty_for_the_default_fold():
        def fold(state, event):
            return state

        assert _fold_service_params(fold) == ()

    def it_returns_the_name_of_a_required_third_parameter():
        def fold(state, event, language):
            return state

        assert _fold_service_params(fold) == ("language",)

    def it_skips_a_third_parameter_that_has_a_default():
        def fold(state, event, language="en"):
            return state

        assert _fold_service_params(fold) == ()

    def it_returns_empty_for_an_unreadable_signature():
        assert _fold_service_params(str) == ()


def describe_build_check():
    def when_a_reducer_service_parameter_is_not_a_registered_service():
        def it_raises_naming_the_reducer_the_parameter_and_known_services():
            reducer = _make_reducer(Reducer)

            with pytest.raises(TypeError) as info:
                EventGraph([_noop], reducers=[reducer], services={"other": "x"})

            message = str(info.value)
            assert "Reducer 'notes' fn parameter 'language' is not a key of " in message
            assert "Known services: ['other']." in message

    def when_a_reducer_fn_declares_a_config_parameter():
        def it_raises_the_same_unknown_service_error():
            def fn(event, config):
                return [config]

            reducer = Reducer(name="notes", event_type=MessageReceived, fn=fn)

            with pytest.raises(TypeError) as info:
                EventGraph([_noop], reducers=[reducer], services={"other": "x"})

            assert "Reducer 'notes' fn parameter 'config' is not a key of " in str(
                info.value
            )

    def when_the_sequence_form_of_services_is_in_use():
        def it_adds_the_mapping_form_sentence():
            reducer = _make_reducer(Reducer)

            with pytest.raises(TypeError) as info:
                EventGraph([_noop], reducers=[reducer], services=["x"])

            assert "services={'name': value}." in str(info.value)

    def when_a_fold_reducer_fold_declares_a_third_required_parameter():
        def it_raises():
            def fold(state, event, language):
                return state

            reducer = FoldReducer(
                name="count",
                event_type=MessageReceived,
                default_factory=dict,
                fold=fold,
            )

            with pytest.raises(TypeError) as info:
                EventGraph([_noop], reducers=[reducer])

            message = str(info.value)
            assert "FoldReducer 'count' fold declares the parameter 'language'." in (
                message
            )
            assert "Use a Reducer or a ScalarReducer." in message

    def when_a_reducer_fn_parameter_has_the_wrong_declared_type():
        def it_raises_a_message_prefixed_by_the_reducer_fn_label():
            def fn(event, language: int):
                return [language]

            reducer = Reducer(name="messages", event_type=MessageReceived, fn=fn)

            with pytest.raises(TypeError) as info:
                EventGraph([_noop], reducers=[reducer], services={"language": "fr"})

            assert str(info.value).startswith("Reducer 'messages' fn")

    def when_a_run_scoped_factory_returns_the_wrong_type():
        def it_raises():
            def factory(config: RunnableConfig) -> int:
                return 1

            def fn(event, language: str):
                return [language]

            reducer = Reducer(name="messages", event_type=MessageReceived, fn=fn)

            with pytest.raises(TypeError) as info:
                EventGraph(
                    [_noop],
                    reducers=[reducer],
                    services={"language": RunScoped(factory)},
                )

            assert str(info.value).startswith("Reducer 'messages' fn")

    def when_fn_has_an_unreadable_signature():
        def it_builds():
            reducer = ScalarReducer(name="text", event_type=MessageReceived, fn=str)

            EventGraph([_noop], reducers=[reducer])

    def when_a_namespace_class_attribute_reducer_has_a_service_parameter():
        def it_builds():
            EventGraph([_NamespaceWithService.Act], services={"language": "fr"})


def describe_reducers_for():
    def when_no_reducer_declares_a_service_parameter():
        def it_returns_the_graphs_reducers_itself():
            reducer = Reducer(name="notes", event_type=MessageReceived, fn=lambda e: [])
            graph = EventGraph([_noop], reducers=[reducer])

            result = graph._reducers_for(None)

            assert result is graph._reducers

    def when_a_reducer_declares_a_service_parameter():
        def it_returns_bound_copies():
            reducer = _make_reducer(Reducer)
            graph = EventGraph([_noop], reducers=[reducer], services={"language": "fr"})

            result = graph._reducers_for(None)

            assert result is not graph._reducers
            assert result["notes"].collect([MessageReceived(text="hi")]) == ["fr:hi"]


def describe_reflect():
    def when_a_reducer_declares_a_service_parameter():
        def it_uses_the_configs_bound_value():
            reducer = _make_reducer(Reducer)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
            )
            log = EventLog([MessageReceived(text="hi")])
            config = {"configurable": {"language": "fr"}}

            reflection = graph.reflect(log, config=config)

            assert reflection.state() == {"notes": ["fr:hi"]}

    def when_config_is_omitted_and_the_factory_reads_a_missing_key():
        def it_raises_the_factorys_key_error():
            reducer = _make_reducer(Reducer)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
            )
            log = EventLog([MessageReceived(text="hi")])

            with pytest.raises(KeyError, match="language"):
                graph.reflect(log).state()


def describe_node_paths_use_bound_reducers():
    """``EventGraph.invoke`` / ``.ainvoke`` project events through bound reducers."""

    def when_a_seed_event_matches_a_service_bound_reducer():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_projects_the_bound_value_into_the_channel(reducer_cls):
            reducer = _make_reducer(reducer_cls)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "seed-sync", "language": "fr"}}

            graph.invoke(MessageReceived(text="hi"), config=config)

            value = graph.compiled.get_state(config).values["notes"]
            assert _collected_value(reducer_cls, value) == "fr:hi"

        async def it_projects_the_bound_value_through_ainvoke():
            reducer = _make_reducer(Reducer)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "seed-async", "language": "fr"}}

            await graph.ainvoke(MessageReceived(text="hi"), config=config)

            snapshot = await graph.compiled.aget_state(config)
            assert snapshot.values["notes"] == ["fr:hi"]

    def when_a_handler_produced_event_matches_a_service_bound_reducer():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_projects_the_bound_value_into_the_channel(reducer_cls):
            reducer = _make_reducer(reducer_cls)
            graph = EventGraph(
                [_relay],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "handler-sync", "language": "fr"}}

            graph.invoke(Pinged(), config=config)

            value = graph.compiled.get_state(config).values["notes"]
            assert _collected_value(reducer_cls, value) == "fr:hi"

    def when_a_second_run_on_one_thread_follows_a_first():
        @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
        def it_adds_a_bound_contribution(reducer_cls):
            reducer = _make_reducer(reducer_cls)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "second-run", "language": "fr"}}

            graph.invoke(MessageReceived(text="hi"), config=config)
            graph.invoke(MessageReceived(text="there"), config=config)

            value = graph.compiled.get_state(config).values["notes"]
            if reducer_cls is Reducer:
                assert value == ["fr:hi", "fr:there"]
            else:
                assert value == "fr:there"

    def when_a_handler_declares_a_reflection_parameter():
        def it_lets_state_reflect_the_bound_value():
            _reflection_states.clear()
            reducer = _make_reducer(Reducer)
            graph = EventGraph(
                [_capture_reflection_state],
                reducers=[reducer],
                services={"language": RunScoped(_language_for)},
            )

            graph.invoke(
                MessageReceived(text="hi"),
                config={"configurable": {"language": "fr"}},
            )

            assert _reflection_states == [{"notes": ["fr:hi"]}]

    def when_a_plain_service_is_registered():
        def it_binds_the_plain_value():
            reducer = _make_reducer(Reducer)
            graph = EventGraph(
                [_noop],
                reducers=[reducer],
                services={"language": "fr"},
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "plain-service"}}

            graph.invoke(MessageReceived(text="hi"), config=config)

            value = graph.compiled.get_state(config).values["notes"]
            assert value == ["fr:hi"]


async def _stream_notes(factory: Any, config: dict[str, Any]) -> tuple[list, dict]:
    """Stream ``Pinged`` through a graph whose reducer needs a ``language``.

    ``include_llm_tokens=True`` selects the shadow stream path. That path
    keeps its own reducer state and calls the reducer ``fn`` directly.
    ``_relay`` turns the seed into a ``MessageReceived``, so the reducer
    projects a handler-produced event.

    Clears ``_factory_configs`` first. This keeps the count a caller reads
    from it independent of any earlier test that also called this helper.

    Returns the stream frames and the checkpoint values.
    """
    _factory_configs.clear()
    graph = EventGraph(
        [_relay],
        reducers=[_make_reducer(Reducer)],
        services={"language": RunScoped(factory)},
        checkpointer=MemorySaver(),
    )
    frames = [
        item
        async for item in graph.astream_events(
            Pinged(),
            include_reducers=True,
            include_llm_tokens=True,
            config=config,
        )
    ]
    snapshot = await graph.compiled.aget_state(config)
    return frames, snapshot.values


def describe_stream_path_uses_bound_reducers():
    """``EventGraph.astream_events`` projects events through bound reducers."""

    def when_a_handler_produced_event_matches_a_service_bound_reducer():
        async def it_streams_the_checkpoint_value():
            config = {"configurable": {"thread_id": "stream-equal", "language": "fr"}}

            frames, values = await _stream_notes(_recording_language_for, config)

            assert isinstance(frames[-1].event, MessageReceived)
            assert frames[-1].reducers["notes"] == values["notes"]
            assert values["notes"] == ["fr:hi"]

        async def it_gives_the_factory_the_same_config_on_every_path():
            config = {"configurable": {"thread_id": "stream-config", "language": "fr"}}
            expected = {
                "configurable": {"thread_id": "stream-config", "language": "fr"}
            }

            await _stream_notes(_recording_language_for, config)

            # One factory call per bind site: the seed node, the handler
            # node, and the stream's own shadow-reducer bind.
            assert len(_factory_configs) == 3
            assert all(seen == expected for seen in _factory_configs)

    def when_the_factory_reads_top_level_metadata():
        async def it_receives_none_on_every_path():
            _metadata_seen.clear()
            config = {
                "configurable": {"thread_id": "stream-metadata"},
                "metadata": {"a": 1},
            }

            await _stream_notes(_metadata_probe, config)

            # One factory call per bind site: the seed node, the handler
            # node, and the stream's own shadow-reducer bind.
            assert len(_metadata_seen) == 3
            assert all(seen is None for seen in _metadata_seen)

    def when_the_stream_call_has_no_config():
        async def it_raises_the_key_error_of_the_factory():
            graph = EventGraph(
                [_noop],
                reducers=[_make_reducer(Reducer)],
                services={"language": RunScoped(_language_for)},
            )

            with pytest.raises(KeyError, match="language"):
                async for _ in graph.astream_events(
                    MessageReceived(text="hi"),
                    include_reducers=True,
                    include_llm_tokens=True,
                ):
                    pass


def describe_agui_adapter_uses_bound_reducers():
    """``AGUIAdapter`` computes resume channel updates with bound reducers."""

    def when_a_frontend_state_event_matches_a_service_bound_reducer():
        def it_returns_a_bound_contribution():
            reducer = Reducer(
                name="notes", event_type=FrontendStateMutated, fn=_project_focus
            )
            graph = EventGraph(
                [_noop],
                reducers=[message_reducer(), reducer],
                services={"language": RunScoped(_language_for)},
            )
            adapter = AGUIAdapter(graph=graph, seed_factory=lambda inp: Pinged())
            config = {"configurable": {"thread_id": "adapter", "language": "fr"}}

            updates = adapter._reducer_updates_for(
                [FrontendStateMutated(state={"focus": "scene"})], config
            )

            assert updates["notes"] == ["fr:scene"]


def describe_replay_reducer():
    def when_the_reducer_has_no_service_parameter():
        def when_services_is_absent():
            def it_folds_the_events():
                from langgraph_events.serde import replay_reducer

                reducer = Reducer(
                    name="notes", event_type=MessageReceived, fn=lambda e: [e.text]
                )
                events = [MessageReceived(text="hi")]

                assert replay_reducer(reducer, events) == ["hi"]

        def when_services_is_present():
            def it_folds_the_events_and_ignores_the_extra_key():
                from langgraph_events.serde import replay_reducer

                reducer = Reducer(
                    name="notes", event_type=MessageReceived, fn=lambda e: [e.text]
                )
                events = [MessageReceived(text="hi")]

                result = replay_reducer(reducer, events, services={"language": "fr"})

                assert result == ["hi"]

    def when_the_reducer_has_a_service_parameter():
        def when_services_is_given():
            @pytest.mark.parametrize("reducer_cls", [Reducer, ScalarReducer])
            def it_binds_the_plain_service_value(reducer_cls):
                from langgraph_events.serde import replay_reducer

                reducer = _make_reducer(reducer_cls)
                events = [MessageReceived(text="hi")]

                result = replay_reducer(reducer, events, services={"language": "fr"})

                assert _collected_value(reducer_cls, result) == "fr:hi"

        def when_services_is_not_given():
            def it_raises_the_unbound_guard_type_error():
                from langgraph_events.serde import replay_reducer

                reducer = _make_reducer(Reducer)
                events = [MessageReceived(text="hi")]

                with pytest.raises(TypeError, match="notes") as info:
                    replay_reducer(reducer, events)

                assert "language" in str(info.value)

        def when_services_holds_a_run_scoped_value():
            def it_raises_naming_run_scoped():
                from langgraph_events.serde import replay_reducer

                reducer = _make_reducer(Reducer)
                events = [MessageReceived(text="hi")]

                with pytest.raises(TypeError, match="RunScoped"):
                    replay_reducer(
                        reducer,
                        events,
                        services={"language": RunScoped(_language_for)},
                    )

        def when_services_is_missing_the_needed_key():
            def it_raises_naming_the_reducer_and_the_service():
                from langgraph_events.serde import replay_reducer

                reducer = _make_reducer(Reducer)
                events = [MessageReceived(text="hi")]

                with pytest.raises(TypeError, match="notes") as info:
                    replay_reducer(reducer, events, services={})

                assert "language" in str(info.value)
