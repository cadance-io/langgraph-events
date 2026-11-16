"""The strict parameter rules that handlers, factories and ``Derived`` share."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 - a factory annotation resolves at graph build
)

from langgraph_events import (
    Derived,
    EventGraph,
    IntegrationEvent,
    Reducer,
    ReducerNotSetError,
    RunScoped,
    ScalarReducer,
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


word = ScalarReducer(name="word", event_type=Told, fn=_word_of)
flags = Reducer(name="flags", event_type=Told, fn=_flags_of)


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
                EventGraph([_unannotated_handler()], services={"setting": "on"})

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
                EventGraph([use], services={"tool": RunScoped(tool_for)})

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
                EventGraph([_unresolved_handler()], services={"setting": "on"})

    def when_it_is_a_factory_parameter():
        def it_raises_naming_the_cause():
            def tool_for(config: Missing) -> Tool:  # type: ignore[name-defined]  # noqa: F821
                return Tool()

            @on(Asked)
            def use(event: Asked, tool: Tool) -> None:
                pass

            with pytest.raises(TypeError, match=r"parameter 'config' " + _UNRESOLVED):
                EventGraph([use], services={"tool": RunScoped(tool_for)})

    def when_it_is_a_derived_parameter():
        def it_raises_naming_the_cause():
            def loud_of(word: Missing) -> str:  # type: ignore[name-defined]  # noqa: F821
                return str(word)

            loud = Derived(name="loud", fn=loud_of)

            with pytest.raises(TypeError, match=r"parameter 'word' " + _UNRESOLVED):
                EventGraph([_shown_handler()], reducers=[word, loud])


def describe_the_numeric_tower():
    def when_the_annotation_is_float_at_the_top_level():
        def it_accepts_an_int_value():
            def count_of(e: Told) -> int:
                return len(e.word)

            count = ScalarReducer(name="count", event_type=Told, fn=count_of)

            @on(Asked)
            def show(event: Asked, count: float = 0.0) -> None:
                pass

            EventGraph([show], reducers=[count])

    def when_the_annotation_nests_float_in_a_list():
        def it_refuses_a_list_of_int():
            def counts_of(e: Told) -> list[int]:
                return [len(e.word)]

            counts = Reducer(name="counts", event_type=Told, fn=counts_of)

            @on(Asked)
            def show(event: Asked, counts: list[float]) -> None:
                pass

            with pytest.raises(
                TypeError, match=r"reducer 'counts' gives list\[int\]\. Change"
            ):
                EventGraph([show], reducers=[counts])


def describe_an_invariant_container():
    def when_the_item_type_is_a_subtype():
        def it_refuses_the_annotation_as_mypy_does():
            @on(Asked)
            def show(event: Asked, flags: list[int]) -> None:
                pass

            with pytest.raises(
                TypeError, match=r"reducer 'flags' gives list\[bool\]\. Change"
            ):
                EventGraph([show], reducers=[flags])


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
                def show(event: Asked, echo: str | None) -> None:
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
