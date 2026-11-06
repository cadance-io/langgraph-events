"""``EventGraph(context_schema=...)``: a typed run context, never checkpointed."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from ag_ui.core import EventType, RunAgentInput
from conftest import Ended, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    EventGraph,
    IntegrationEvent,
    Interrupted,
    RunScoped,
    message_reducer,
    on,
)
from langgraph_events.agui import AGUIAdapter

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


def _drain(items: Iterable[object]) -> None:
    for _ in items:
        pass


async def _adrain(items: AsyncIterator[object] | Awaitable[object]) -> None:
    if isinstance(items, AsyncIterator):
        async for _ in items:
            pass
    else:
        await items


_SYNC_STARTS: dict[str, Callable[[EventGraph], object]] = {
    "invoke": lambda g: g.invoke(Started(), config=_thread("n")),
    "stream_events": lambda g: _drain(g.stream_events(Started(), config=_thread("n"))),
}
_ASYNC_STARTS: dict[str, Callable[[EventGraph], Any]] = {
    "ainvoke": lambda g: g.ainvoke(Started(), config=_thread("n")),
    "astream_events": lambda g: g.astream_events(Started(), config=_thread("n")),
}
_SYNC_RESUMES: dict[str, Callable[[EventGraph], object]] = {
    "resume": lambda g: g.resume(Answered(), config=_thread("t")),
    "stream_resume": lambda g: _drain(g.stream_resume(Answered(), config=_thread("t"))),
}
_ASYNC_RESUMES: dict[str, Callable[[EventGraph], Any]] = {
    "aresume": lambda g: g.aresume(Answered(), config=_thread("t")),
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

            def when_the_run_is_async():
                async def it_injects_the_run_context():
                    graph = EventGraph([greet], context_schema=RunFacts)

                    log = await graph.ainvoke(Started(), context=RunFacts("de"))

                    assert log.latest(Ended) == Ended(result="de")

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
            def it_leaves_the_parameter_unclaimed():
                with pytest.raises(TypeError, match="cannot inject"):
                    EventGraph([greet])

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
            @pytest.mark.parametrize("start", _SYNC_STARTS.values(), ids=_SYNC_STARTS)
            def it_raises_naming_the_schema(start: Callable[[EventGraph], object]):
                graph = EventGraph([greet], context_schema=RunFacts)

                with pytest.raises(TypeError, match="RunFacts"):
                    start(graph)

            def when_the_run_is_async():
                @pytest.mark.parametrize(
                    "start", _ASYNC_STARTS.values(), ids=_ASYNC_STARTS
                )
                async def it_raises_naming_the_schema(
                    start: Callable[[EventGraph], Any],
                ):
                    graph = EventGraph([greet], context_schema=RunFacts)

                    with pytest.raises(TypeError, match="RunFacts"):
                        await _adrain(start(graph))

        def when_a_thread_resumes():
            @pytest.mark.parametrize(
                "resume", _SYNC_RESUMES.values(), ids=_SYNC_RESUMES
            )
            def it_raises_naming_the_schema(resume: Callable[[EventGraph], object]):
                graph = _paused_graph()

                with pytest.raises(TypeError, match="RunFacts"):
                    resume(graph)

                assert graph.get_state(_thread("t")).is_interrupted

            def when_the_resume_is_async():
                @pytest.mark.parametrize(
                    "resume", _ASYNC_RESUMES.values(), ids=_ASYNC_RESUMES
                )
                async def it_raises_naming_the_schema(
                    resume: Callable[[EventGraph], Any],
                ):
                    graph = _paused_graph()

                    with pytest.raises(TypeError, match="RunFacts"):
                        await _adrain(resume(graph))


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
