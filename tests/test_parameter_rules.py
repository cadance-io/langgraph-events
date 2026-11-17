"""The strict parameter rules that handlers, factories and ``Derived`` share."""

import functools
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import services_from
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 - a factory annotation resolves at graph build
)

from langgraph_events import (
    SKIP,
    Derived,
    EventGraph,
    IntegrationEvent,
    Reducer,
    ReducerNotSetError,
    RunScoped,
    ScalarReducer,
    SkipType,
    on,
)


class Asked(IntegrationEvent):
    pass


class Told(IntegrationEvent):
    word: str = ""


class Counted(IntegrationEvent):
    n: int = 0


class Shown(IntegrationEvent):
    text: str = ""


def _word_of(e: Told) -> str:
    return e.word


def _flags_of(e: Told) -> list[bool]:
    return [bool(e.word)]


word = ScalarReducer[Told, str](name="word", event_type=Told, fn=_word_of)
flags = Reducer[Told, bool](name="flags", event_type=Told, fn=_flags_of)


class Tool:
    """A per-run value that a factory builds."""


def _tool_for(config: RunnableConfig) -> Tool:
    return Tool()


_UNRESOLVED = (
    r"does not resolve \(NameError: .*Missing.*\)\. Make every name in it "
    r"importable at run time\. A usual cause is an import under 'if "
    r"TYPE_CHECKING:'"
)


def _unannotated_handler() -> Any:
    @on(Asked)
    def show(event: Asked, setting) -> None:  # type: ignore[no-untyped-def]
        pass

    return show


def _unresolved_handler() -> Any:
    @on(Asked)
    def show(event: Asked, setting: Missing) -> None:  # type: ignore[name-defined]  # noqa: F821
        pass

    return show


def describe_an_unannotated_parameter():
    def when_it_is_a_handler_parameter():
        def it_raises_naming_the_handler_the_parameter_and_the_fix():
            with pytest.raises(
                TypeError,
                match=r"Handler '.*show' \(node 'show'\) has no annotation on "
                r"parameter 'setting'\. Annotate it with the type of its input",
            ):
                EventGraph(
                    [_unannotated_handler()], services=services_from({"setting": "on"})
                )

    def when_it_is_a_factory_parameter():
        def it_raises_naming_the_factory_the_parameter_and_the_fix():
            def tool_for(config) -> Tool:  # type: ignore[no-untyped-def]
                return Tool()

            @on(Asked)
            def use(event: Asked, tool: Tool) -> None:
                pass

            with pytest.raises(
                TypeError,
                match=r"RunScoped factory '.*tool_for' for service 'tool' has no "
                r"annotation on parameter 'config'\. Annotate it",
            ):
                EventGraph([use], services=services_from({"tool": RunScoped(tool_for)}))

    def when_it_is_a_derived_parameter():
        def it_raises_naming_the_derived_the_parameter_and_the_fix():
            def loud_of(word) -> str:  # type: ignore[no-untyped-def]
                return str(word).upper()

            loud = Derived(name="loud", fn=loud_of)

            with pytest.raises(
                TypeError,
                match=r"Derived 'loud' has no annotation on parameter 'word'\. "
                r"Annotate it",
            ):
                EventGraph([_shown_handler()], reducers=[word, loud])


def _shown_handler() -> Any:
    @on(Asked)
    def show(event: Asked) -> Shown:
        return Shown()

    return show


def describe_a_parameter_annotation_that_does_not_resolve():
    def when_it_is_a_handler_parameter():
        def it_raises_naming_the_cause():
            with pytest.raises(TypeError, match=r"parameter 'setting' " + _UNRESOLVED):
                EventGraph(
                    [_unresolved_handler()], services=services_from({"setting": "on"})
                )

    def when_it_is_a_factory_parameter():
        def it_raises_naming_the_cause():
            def tool_for(config: Missing) -> Tool:  # type: ignore[name-defined]  # noqa: F821
                return Tool()

            @on(Asked)
            def use(event: Asked, tool: Tool) -> None:
                pass

            with pytest.raises(TypeError, match=r"parameter 'config' " + _UNRESOLVED):
                EventGraph([use], services=services_from({"tool": RunScoped(tool_for)}))

    def when_it_is_a_derived_parameter():
        def it_raises_naming_the_cause():
            def loud_of(word: Missing) -> str:  # type: ignore[name-defined]  # noqa: F821
                return str(word)

            loud = Derived(name="loud", fn=loud_of)

            with pytest.raises(TypeError, match=r"parameter 'word' " + _UNRESOLVED):
                EventGraph([_shown_handler()], reducers=[word, loud])


def describe_a_derived_parameter():
    def when_the_annotation_does_not_fit_the_input():
        def it_raises_at_graph_build():
            def size_of(word: int) -> int:
                return word

            size = Derived(name="size", fn=size_of)

            with pytest.raises(
                TypeError,
                match=r"Derived 'size' declares parameter 'word' as int, but "
                r"reducer 'word' gives str \| None",
            ):
                EventGraph([_shown_handler()], reducers=[word, size])

    def when_the_parameter_has_a_default():
        def when_the_input_is_unset():
            def it_receives_the_default():
                def echo_of(word: str = "quiet") -> str:
                    return word

                echo = Derived(name="echo", fn=echo_of)
                seen: list[str] = []

                @on(Asked)
                def show(event: Asked, echo: str) -> None:
                    seen.append(echo)

                EventGraph([show], reducers=[word, echo]).invoke(Asked())

                assert seen == ["quiet"]

    def when_the_annotation_rejects_none():
        def when_the_input_is_unset():
            def it_raises_reducer_not_set_error_naming_the_derived():
                def echo_of(word: str) -> str:
                    return word

                echo = Derived(name="echo", fn=echo_of)

                @on(Asked)
                def show(event: Asked, echo: str) -> None:
                    pass

                graph = EventGraph([show], reducers=[word, echo])

                with pytest.raises(
                    ReducerNotSetError,
                    match=r"Derived 'echo' requires reducer 'word' to be set.*"
                    r"before this Derived runs",
                ):
                    graph.invoke(Asked())

    def when_the_annotation_names_the_run_config():
        def it_raises_because_a_derived_reads_no_run_input():
            def config_of(config: RunnableConfig) -> str:
                return ""

            echo = Derived(name="echo", fn=config_of)

            with pytest.raises(
                TypeError,
                match=r"Derived 'echo' reads the config in parameter 'config'\. A "
                r"Derived reads only stored reducers, Derived values and the "
                r"EventLog",
            ):
                EventGraph([_shown_handler()], reducers=[word, echo])


def _word_or_skip(e: Told) -> str | SkipType:
    return e.word or SKIP


def describe_a_scalar_reducer_that_can_skip():
    def it_leaves_skip_out_of_the_stored_type():
        maybe = ScalarReducer[Told, str](
            name="maybe", event_type=Told, fn=_word_or_skip
        )

        @on(Asked)
        def show(event: Asked, maybe: str | None) -> None:
            pass

        EventGraph([show], reducers=[maybe])


def _scaled(word: str | None, factor: int) -> str:
    return (word or "") * factor


def describe_a_derived_from_a_partial():
    def when_the_partial_binds_a_keyword():
        def it_binds_only_the_parameters_that_the_partial_leaves_open():
            loud = Derived(name="loud", fn=functools.partial(_scaled, factor=2))
            seen: list[str] = []

            @on(Asked)
            def show(event: Asked, loud: str) -> None:
                seen.append(loud)

            EventGraph([show], reducers=[word, loud]).invoke([Told(word="a"), Asked()])

            assert seen == ["aa"]


@dataclass(frozen=True)
class Facts:
    language: str


def describe_a_derived_parameter_that_names_the_context_schema():
    def it_raises_because_a_derived_reads_no_run_input():
        def facts_of(facts: Facts) -> str:
            return facts.language

        echo = Derived(name="echo", fn=facts_of)

        with pytest.raises(
            TypeError,
            match=r"Derived 'echo' reads the context in parameter 'facts'\. A "
            r"Derived reads only",
        ):
            EventGraph([_shown_handler()], reducers=[word, echo], context_schema=Facts)


@dataclass(frozen=True)
class Summary:
    word: str | None


summary = Derived(name="summary", fn=Summary)


def describe_a_derived_whose_function_is_a_class():
    def when_a_reader_names_another_type():
        def it_raises_naming_the_class():
            @on(Asked)
            def show(event: Asked, summary: int) -> None:
                pass

            with pytest.raises(
                TypeError, match=r"Derived 'summary' gives Summary\. Change"
            ):
                EventGraph([show], reducers=[word, summary])

    def when_a_reader_names_the_class():
        def it_gives_an_instance():
            seen: list[object] = []

            @on(Asked)
            def show(event: Asked, summary: Summary) -> None:
                seen.append(summary)

            EventGraph([show], reducers=[word, summary]).invoke(
                [Told(word="a"), Asked()]
            )

            assert seen == [Summary(word="a")]


def _only_skip(e: Told) -> SkipType:
    return SKIP


def describe_a_scalar_reducer_that_only_skips():
    def it_raises_naming_the_reducer_and_the_fix():
        never = ScalarReducer[Told, Any](name="never", event_type=Told, fn=_only_skip)

        @on(Asked)
        def show(event: Asked, never: str | None) -> None:
            pass

        with pytest.raises(
            TypeError,
            match=r"ScalarReducer 'never' has a fn that returns only SkipType, so "
            r"it never sets a value\.",
        ):
            EventGraph([show], reducers=[never])
