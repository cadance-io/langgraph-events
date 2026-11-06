"""``EventGraph(context_schema=...)``: a typed run context, never checkpointed."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from ag_ui.core import EventType, RunAgentInput
from conftest import Ended, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    EventGraph,
    EventStore,
    EventTopology,
    IntegrationEvent,
    Interrupted,
    RunScoped,
    message_reducer,
    on,
)
from langgraph_events.agui import AGUIAdapter
from langgraph_events.serde import assert_resume_recovers

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@dataclass(frozen=True)
class RunFacts:
    """The per-run facts a server sets, such as the run language."""

    language: str


class _Model:
    def __init__(self, language: str, thread_id: str) -> None:
        self.language = language
        self.thread_id = thread_id


def _model_for(config: RunnableConfig, context: RunFacts) -> _Model:
    return _Model(context.language, config["configurable"]["thread_id"])


def _optional_context(
    config: RunnableConfig, context: RunFacts | None = None
) -> _Model:
    return _Model("", "")


def _keyword_context(config: RunnableConfig, *, context: RunFacts) -> _Model:
    return _Model("", "")


def _other_context(config: RunnableConfig, context: str) -> _Model:
    return _Model("", "")


def _config_only(config: RunnableConfig, retries: int = 3) -> _Model:
    return _Model("none", config["configurable"]["thread_id"])


@dataclass(frozen=True)
class LocalFacts(RunFacts):
    """A subclass of the schema."""


class AnswerAsked(Interrupted):
    language: str

    def agui_dict(self) -> dict[str, Any]:
        return {}


class Answered(IntegrationEvent):
    pass


@on(Started)
def greet(event: Started, facts: RunFacts) -> Ended:
    return Ended(result=facts.language)


@on(Started)
def read_only(event: Started, facts: RunFacts) -> None:
    pass


@on(Started)
def ask(event: Started, facts: RunFacts) -> AnswerAsked:
    return AnswerAsked(language=facts.language)


@on(Answered)
def note_answer(event: Answered, facts: RunFacts) -> Ended:
    return Ended(result=facts.language)


@on(Started)
def maybe_greet(event: Started, facts: RunFacts | None) -> None:
    pass


@on(Started)
def local_greet(event: Started, facts: LocalFacts) -> None:
    pass


@on(Started)
def model_reply(event: Started, model: _Model) -> Ended:
    return Ended(result=f"{model.language}@{model.thread_id}")


def _thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _paused_graph() -> EventGraph:
    graph = EventGraph(
        [ask, note_answer], context_schema=RunFacts, checkpointer=MemorySaver()
    )
    graph.invoke(Started(), config=_thread("t"), context=RunFacts("fr"))
    return graph


async def _finish(run: object) -> None:
    if isinstance(run, AsyncIterator):
        async for _ in run:
            pass
    elif isinstance(run, Awaitable):
        await run


_STARTS: dict[str, Callable[[EventGraph], object]] = {
    "invoke": lambda g: g.invoke(Started(), config=_thread("n")),
    "ainvoke": lambda g: g.ainvoke(Started(), config=_thread("n")),
    "stream_events": lambda g: list(g.stream_events(Started(), config=_thread("n"))),
    "astream_events": lambda g: g.astream_events(Started(), config=_thread("n")),
}
_RESUMES: dict[str, Callable[[EventGraph], object]] = {
    "resume": lambda g: g.resume(Answered(), config=_thread("t")),
    "aresume": lambda g: g.aresume(Answered(), config=_thread("t")),
    "stream_resume": lambda g: list(g.stream_resume(Answered(), config=_thread("t"))),
    "astream_resume": lambda g: g.astream_resume(Answered(), config=_thread("t")),
}


def _input(*, resume: bool = False) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={"command": {"resume": "yes"}} if resume else {},
    )


def _adapter(graph: EventGraph) -> AGUIAdapter:
    return AGUIAdapter(
        graph=graph,
        seed_factory=lambda inp: Started(),
        resume_factory=lambda inp: Answered() if inp.forwarded_props else None,
        on_unmapped="ignore",
    )


async def _stream(
    adapter: AGUIAdapter, inp: RunAgentInput, facts: RunFacts
) -> list[Any]:
    return [e async for e in adapter.stream(inp, context=facts)]


def describe_context_schema():
    def describe_handler_injection():
        def when_a_parameter_has_the_schema_annotation():
            def it_injects_the_run_context():
                graph = EventGraph([greet], context_schema=RunFacts)

                log = graph.invoke(Started(), context=RunFacts("fr"))

                assert log.latest(Ended) == Ended(result="fr")

            def when_the_context_is_a_dict():
                def it_injects_the_context_coerced_to_the_schema():
                    graph = EventGraph([greet], context_schema=RunFacts)

                    log = graph.invoke(Started(), context={"language": "it"})

                    assert log.latest(Ended) == Ended(result="it")

        def when_a_thread_resumes():
            def it_injects_the_context_of_the_resume():
                graph = _paused_graph()

                log = graph.resume(
                    Answered(), config=_thread("t"), context=RunFacts("es")
                )

                assert log.latest(Ended) == Ended(result="es")

        def when_a_caller_runs_the_compiled_graph_directly():
            def without_a_context():
                def it_raises_naming_the_handler():
                    graph = EventGraph([greet], context_schema=RunFacts)

                    with pytest.raises(ValueError, match="'greet'"):
                        graph.compiled.invoke({"events": [Started()]})

        def without_a_context_schema():
            def it_names_context_schema_in_the_error():
                with pytest.raises(TypeError) as info:
                    EventGraph([greet])

                assert "cannot inject" in str(info.value)
                assert (
                    "If 'facts' is the run context, pass context_schema=RunFacts."
                    in str(info.value)
                )

        def when_a_parameter_is_the_optional_schema():
            def it_says_only_the_exact_type_matches():
                with pytest.raises(TypeError) as info:
                    EventGraph([maybe_greet], context_schema=RunFacts)

                assert (
                    "'facts' is annotated test_run_context.RunFacts | None, but "
                    "only the exact type context_schema=RunFacts receives the run "
                    "context." in str(info.value)
                )

        def when_a_parameter_is_a_subclass_of_the_schema():
            def it_says_only_the_exact_type_matches():
                with pytest.raises(TypeError) as info:
                    EventGraph([local_greet], context_schema=RunFacts)

                assert (
                    "'facts' is annotated LocalFacts, but only the exact type "
                    "context_schema=RunFacts receives" in str(info.value)
                )

    def describe_EventTopology():
        def it_does_not_list_the_context_parameter_as_a_requirement():
            topology = EventTopology([greet], context_schema=RunFacts)

            assert topology.requirements == ()
            assert topology.context_schema is RunFacts

        def it_gives_the_schema_to_EventGraph_of():
            graph = EventGraph.of(EventTopology([greet], context_schema=RunFacts))

            log = graph.invoke(Started(), context=RunFacts("de"))

            assert log.latest(Ended) == Ended(result="de")

        def it_builds_a_store_that_needs_no_context():
            saver = MemorySaver()
            _ = EventGraph(
                [ask, note_answer], context_schema=RunFacts, checkpointer=saver
            ).invoke(Started(), config=_thread("s"), context=RunFacts("fr"))
            store = EventStore(
                EventTopology([ask, note_answer], context_schema=RunFacts), saver
            )

            assert store.threads_paused_on(thread_ids=["s"]) == [_thread("s")]

    def describe_RunScoped():
        def when_the_factory_takes_two_parameters():
            def it_receives_the_run_config_and_the_run_context():
                graph = EventGraph(
                    [model_reply],
                    context_schema=RunFacts,
                    services={"model": RunScoped(_model_for)},
                )

                log = graph.invoke(
                    Started(), config=_thread("t1"), context=RunFacts("fr")
                )

                assert log.latest(Ended) == Ended(result="fr@t1")

            def without_a_context_schema():
                def it_raises_at_build():
                    with pytest.raises(TypeError, match="context_schema") as info:
                        EventGraph(
                            [model_reply],
                            services={"model": RunScoped(_model_for)},
                        )

                    assert "'model'" in str(info.value)

        def when_the_second_parameter_is_optional():
            def without_a_context_schema():
                def it_passes_the_run_config_only():
                    graph = EventGraph(
                        [model_reply], services={"model": RunScoped(_config_only)}
                    )

                    log = graph.invoke(Started(), config=_thread("t3"))

                    assert log.latest(Ended) == Ended(result="none@t3")

        def when_the_second_parameter_is_not_the_context():
            @pytest.mark.parametrize(
                ("factory", "why"),
                [
                    (_optional_context, "has an optional parameter 'context'"),
                    (_keyword_context, "has a keyword-only parameter 'context'"),
                    (_other_context, "annotates parameter 'context' as str"),
                ],
                ids=["optional", "keyword-only", "another-type"],
            )
            def it_raises_naming_the_factory_and_the_fix(factory: Any, why: str):
                with pytest.raises(TypeError) as info:
                    EventGraph(
                        [model_reply],
                        context_schema=RunFacts,
                        services={"model": RunScoped(factory)},
                    )

                assert str(info.value) == (
                    f"RunScoped factory {factory.__qualname__!r} for service "
                    f"'model' {why}. A factory that takes the run context "
                    f"declares it as a second positional parameter, required and "
                    f"annotated with RunFacts: (config, context: RunFacts). "
                    f"Change the parameter, or remove it."
                )

        def when_python_cannot_read_the_factory_signature():
            def it_passes_the_run_config_only():
                @on(Started)
                def copy_config(event: Started, copied: dict) -> Ended:  # type: ignore[type-arg]
                    return Ended(result=copied["configurable"]["thread_id"])

                graph = EventGraph(
                    [copy_config],
                    context_schema=RunFacts,
                    services={"copied": RunScoped(dict)},
                )

                log = graph.invoke(
                    Started(), config=_thread("t2"), context=RunFacts("fr")
                )

                assert log.latest(Ended) == Ended(result="t2")

    def describe_type_keyed_services():
        def when_a_service_is_an_instance_of_the_schema():
            def it_raises_at_build():
                with pytest.raises(TypeError) as info:
                    EventGraph(
                        [greet], context_schema=RunFacts, services=[RunFacts("fr")]
                    )

                assert str(info.value).startswith(
                    "services= holds an instance of RunFacts, the context_schema "
                    "of the graph."
                )

    def describe_run_call():
        def when_a_keyword_is_misspelled():
            def it_raises_naming_the_keyword():
                graph = EventGraph([greet], context_schema=RunFacts)

                with pytest.raises(TypeError, match=r"\['contxt'\]"):
                    graph.invoke(Started(), contxt=RunFacts("fr"))

        def when_the_context_has_another_type():
            def it_raises_naming_both_types():
                graph = EventGraph([greet], context_schema=RunFacts)

                with pytest.raises(TypeError) as info:
                    graph.invoke(Started(), context="fr")

                assert str(info.value) == (
                    "context= must be an instance of RunFacts, the context_schema "
                    "of the graph, got str."
                )

    def describe_checkpoint():
        def it_keeps_the_context_out_of_the_metadata_and_the_state():
            saver = MemorySaver()
            graph = EventGraph([read_only], context_schema=RunFacts, checkpointer=saver)
            marker = "context-only-7f3a"

            graph.invoke(Started(), config=_thread("c"), context=RunFacts(marker))

            tuples = list(saver.list(_thread("c")))
            assert tuples
            for tup in tuples:
                assert marker not in repr(tup.metadata)
                assert marker not in repr(tup.checkpoint["channel_values"])
            assert marker not in repr(graph.get_state(_thread("c")))

    def describe_missing_context():
        def when_a_run_starts():
            @pytest.mark.parametrize("start", _STARTS.values(), ids=_STARTS)
            async def it_raises_naming_the_schema(
                start: Callable[[EventGraph], object],
            ):
                graph = EventGraph([greet], context_schema=RunFacts)

                with pytest.raises(TypeError, match="RunFacts"):
                    await _finish(start(graph))

        def when_a_thread_resumes():
            @pytest.mark.parametrize("resume", _RESUMES.values(), ids=_RESUMES)
            async def it_raises_naming_the_schema(
                resume: Callable[[EventGraph], object],
            ):
                graph = _paused_graph()

                with pytest.raises(TypeError, match="RunFacts"):
                    await _finish(resume(graph))

                assert graph.get_state(_thread("t")).is_interrupted

    def describe_assert_resume_recovers():
        def it_passes_the_context_to_both_runs():
            saver = MemorySaver()
            before, after = (
                EventGraph(
                    [ask, note_answer], context_schema=RunFacts, checkpointer=saver
                )
                for _ in range(2)
            )

            log = assert_resume_recovers(
                before,
                after,
                seed=Started(),
                resume_with=Answered(),
                context=RunFacts("fr"),
            )

            assert log.latest(Ended) == Ended(result="fr")


def _chat_graph() -> EventGraph:
    return EventGraph(
        [ask, note_answer],
        context_schema=RunFacts,
        checkpointer=MemorySaver(),
        reducers=[message_reducer()],
    )


def describe_AGUIAdapter_stream():
    def when_a_run_starts():
        async def it_passes_the_context_to_the_run():
            graph = _chat_graph()

            await _stream(_adapter(graph), _input(), RunFacts("fr"))

            assert graph.get_state(_thread("t")).interrupted == AnswerAsked(
                language="fr"
            )

    def when_a_run_resumes():
        async def it_passes_the_context_to_the_resume():
            graph = _chat_graph()
            adapter = _adapter(graph)
            await _stream(adapter, _input(), RunFacts("fr"))

            events = await _stream(adapter, _input(resume=True), RunFacts("es"))

            assert graph.get_state(_thread("t")).events.latest(Ended) == Ended(
                result="es"
            )
            assert events[-1].type == EventType.RUN_FINISHED

    def without_a_context():
        def it_raises_at_the_stream_call():
            graph = _chat_graph()

            with pytest.raises(TypeError, match="RunFacts"):
                _adapter(graph).stream(_input())
