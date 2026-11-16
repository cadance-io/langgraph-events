"""The strict parameter rules that handlers, factories and ``Derived`` share."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pytest
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


def _build_scalar(fn: Any, default: object, annotation: Any) -> None:
    """Build a graph whose handler reads a ScalarReducer of *fn* as *annotation*."""
    scalar = ScalarReducer(name="value", event_type=Told, fn=fn, default=default)

    @on(Asked)
    def show(event: Asked, value: Any) -> None:
        pass

    show.__annotations__["value"] = annotation
    EventGraph([show], reducers=[scalar])


def _tags_of(e: Told) -> tuple[str, ...]:
    return (e.word,)


def _stage_of(e: Told) -> Literal["draft", "sent"]:
    return "sent"


def _names_of(e: Told) -> list[str]:
    return [e.word]


def _size_of(e: Told) -> int:
    return len(e.word)


def _word_or_skip(e: Told) -> str | SkipType:
    return e.word or SKIP


def describe_the_value_type_of_a_scalar_reducer():
    def when_the_default_fits_the_return_type():
        @pytest.mark.parametrize(
            ("fn", "default", "annotation"),
            [
                (_tags_of, (), tuple[str, ...]),
                (_stage_of, "draft", Literal["draft", "sent"]),
                (_names_of, [], list[str]),
            ],
            ids=["tuple", "literal", "list"],
        )
        def it_adds_no_type_for_the_default(fn: Any, default: object, annotation: Any):
            _build_scalar(fn, default, annotation)

    def when_the_default_does_not_fit_the_return_type():
        def it_adds_the_type_of_the_default():
            with pytest.raises(TypeError, match=r"reducer 'value' gives int \| str"):
                _build_scalar(_size_of, "none", int)

    def when_the_function_can_return_skip():
        def it_leaves_skip_out_of_the_value_type():
            _build_scalar(_word_or_skip, None, str | None)


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


def describe_the_fit_of_a_nested_container():
    @pytest.mark.parametrize(
        ("provided", "declared", "fits"),
        [
            (dict[str, list[bool]], Mapping[str, list[int]], False),
            (list[list[bool]], Sequence[list[int]], False),
            (dict[bool, int], Mapping[int, int], False),
            (dict[str, bool], Mapping[str, int], True),
            (dict[str, Any], dict[int, str], False),
            (dict[str, Any], dict[str, int], True),
            (dict, dict[str, int], True),
        ],
        ids=[
            "mapping-value-invariant-list",
            "sequence-of-invariant-list",
            "mapping-key-invariant",
            "mapping-value-covariant",
            "any-value-wrong-key",
            "any-value-right-key",
            "bare-dict",
        ],
    )
    def it_follows_the_variance_of_mypy(provided: Any, declared: Any, fits: bool):
        def items_of(e: Told) -> Any:
            return {}

        items_of.__annotations__["return"] = provided

        @on(Asked)
        def show(event: Asked, items: Any) -> None:
            pass

        show.__annotations__["items"] = declared
        scalar = ScalarReducer(name="items", event_type=Told, fn=items_of)
        if fits:
            EventGraph([show], reducers=[scalar])
        else:
            with pytest.raises(TypeError, match=r"reducer 'items' gives"):
                EventGraph([show], reducers=[scalar])
