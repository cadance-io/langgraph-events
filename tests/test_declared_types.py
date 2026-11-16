"""Declared types: each source declares its type, and a reader names it exactly."""

from __future__ import annotations

import dataclasses
import typing
from collections import abc
from typing import Annotated, Any, Literal, NewType, Optional, TypeVar

import pytest
from langchain_core.messages import BaseMessage  # noqa: TC002 - read at graph build
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 - a factory annotation resolves at graph build
)

from langgraph_events import (
    Derived,
    EventGraph,
    FoldReducer,
    IntegrationEvent,
    Reducer,
    Reflection,
    RunScoped,
    ScalarReducer,
    message_reducer,
    on,
)


class Asked(IntegrationEvent):
    pass


class Set(IntegrationEvent):
    value: Any = None


class Added(IntegrationEvent):
    def fold(self, state: int) -> int:
        return state + 1


UserId = NewType("UserId", int)


def _value_of(e: Set) -> Any:
    return e.value


def _graph(source: object, annotation: object) -> EventGraph:
    """A graph whose handler reads the ScalarReducer of *source* as *annotation*."""
    # A lambda carries no return annotation, so only the declared type counts.
    scalar = ScalarReducer[Set, source](  # type: ignore[valid-type]
        name="value", event_type=Set, fn=lambda e: e.value
    )

    @on(Asked)
    def show(event: Asked, value: Any) -> None:
        pass

    show.__annotations__["value"] = annotation
    return EventGraph([show], reducers=[scalar])


def describe_the_normalization_rule():
    @pytest.mark.parametrize(
        ("source", "reader"),
        [
            (list[int], typing.List[int]),  # noqa: UP006 - the alias is the case
            (list, typing.List),  # noqa: UP006 - the alias is the case
            (abc.Sequence[int], typing.Sequence[int]),
            (abc.Callable[[int], str], typing.Callable[[int], str]),
            (int | None, Optional[int]),  # noqa: UP045 - the spelling is the case
            (Literal["a", "b"], Literal["b", "a"]),
            (Literal["a", "b"], Literal["a"] | Literal["b"]),
            (int, Annotated[int, "meta"]),
            (list[int | None], list[Annotated[int, "meta"] | None]),
            (int | str, str | int),
            (Any, Any),
            (UserId, UserId),
            (int | None, int),
        ],
        ids=[
            "typing-alias",
            "bare-typing-alias",
            "abc-alias",
            "callable-arguments",
            "optional",
            "literal-order",
            "literal-union",
            "annotated",
            "annotated-nested",
            "union-order",
            "any",
            "newtype",
            "required-reader-of-an-optional-reducer",
        ],
    )
    def it_accepts_the_same_type(source: object, reader: object):
        _graph(source, reader)

    @pytest.mark.parametrize(
        ("source", "reader"),
        [
            (Literal[1], Literal[True]),
            (int, Any),
            (Any, int),
            (int, object),
            (int, UserId),
            (bool, int),
            (list[int], abc.Sequence[int]),
            (int, float),
        ],
        ids=[
            "literal-type",
            "any-reader",
            "any-source",
            "object-reader",
            "newtype-reader",
            "subclass",
            "abstract-container",
            "numeric-tower",
        ],
    )
    def it_refuses_another_type(source: object, reader: object):
        with pytest.raises(TypeError, match=r"reducer 'value' gives "):
            _graph(source, reader)


def describe_an_optional_reader():
    def when_the_reducer_has_a_default():
        def it_raises_because_the_value_is_never_none():
            count = ScalarReducer[Set, int](
                name="count", event_type=Set, fn=lambda e: 1, default=0
            )

            @on(Asked)
            def show(event: Asked, count: int | None) -> None:
                pass

            with pytest.raises(TypeError, match=r"reducer 'count' gives int\."):
                EventGraph([show], reducers=[count])


def _word_of(e: Set) -> str:
    return str(e.value)


def describe_a_reducer_type_argument():
    def when_the_reducer_is_not_subscripted():
        def it_raises_naming_the_reducer_and_the_fix():
            bare = ScalarReducer(name="word", event_type=Set, fn=_word_of)

            with pytest.raises(
                TypeError,
                match=r"ScalarReducer 'word' declares no stored type\. Subscript "
                r"it: ScalarReducer\[Set, str\]\(\.\.\.\)",
            ):
                EventGraph([_shown()], reducers=[bare])

    def when_the_type_argument_is_a_string():
        def it_raises_naming_the_fix():
            stringly = ScalarReducer["Set", "str"](  # type: ignore[type-arg]
                name="word", event_type=Set, fn=_word_of
            )

            with pytest.raises(TypeError, match=r"ScalarReducer 'word' declares the "):
                EventGraph([_shown()], reducers=[stringly])

    def when_the_type_argument_is_a_type_variable():
        def it_raises_naming_the_fix():
            free = TypeVar("free")
            generic = ScalarReducer[Set, free](name="word", event_type=Set, fn=_word_of)  # type: ignore[valid-type]

            with pytest.raises(TypeError, match=r"ScalarReducer 'word' declares the "):
                EventGraph([_shown()], reducers=[generic])

    def when_the_function_return_annotation_differs_from_the_type():
        def it_raises_naming_both_types():
            wrong = ScalarReducer[Set, int](name="word", event_type=Set, fn=_word_of)  # type: ignore[arg-type]

            with pytest.raises(
                TypeError,
                match=r"ScalarReducer 'word' declares int, but its fn returns str",
            ):
                EventGraph([_shown()], reducers=[wrong])

    def when_a_dataclass_copy_is_made():
        def it_keeps_the_type():
            scalar = ScalarReducer[Set, str](name="word", event_type=Set, fn=_word_of)
            copy = dataclasses.replace(scalar, name="copy")

            EventGraph([_reads_copy()], reducers=[copy])

    def when_a_subclass_names_the_type():
        def it_reads_the_type_of_the_base():
            class WordReducer(ScalarReducer[Set, str]):
                pass

            EventGraph(
                [_reads_word()],
                reducers=[WordReducer(name="word", event_type=Set, fn=_word_of)],
            )


def _shown() -> Any:
    @on(Asked)
    def show(event: Asked) -> None:
        pass

    return show


def _reads_word() -> Any:
    @on(Asked)
    def show(event: Asked, word: str | None) -> None:
        pass

    return show


def _reads_copy() -> Any:
    @on(Asked)
    def show(event: Asked, copy: str | None) -> None:
        pass

    return show


def _items_of(e: Set) -> list[int]:
    return [1]


def describe_the_stored_type_of_each_reducer():
    def when_it_is_a_list_reducer():
        def it_gives_a_list_of_the_type():
            items = Reducer[Set, int](name="items", event_type=Set, fn=_items_of)

            @on(Asked)
            def show(event: Asked, items: list[int]) -> None:
                pass

            EventGraph([show], reducers=[items])

    def when_it_is_a_fold_reducer():
        def it_gives_the_state_type():
            count = FoldReducer[int](
                name="count", event_type=Added, default_factory=int
            )

            @on(Asked)
            def show(event: Asked, count: int) -> None:
                pass

            EventGraph([show], reducers=[count])

    def when_it_is_the_message_reducer():
        def it_gives_a_list_of_messages():
            @on(Asked)
            def show(event: Asked, messages: list[BaseMessage]) -> None:
                pass

            EventGraph([show], reducers=[message_reducer()])


@dataclasses.dataclass
class Tool:
    name: str = "tool"


@dataclasses.dataclass
class SubTool(Tool):
    pass


def _tool_for(config: RunnableConfig) -> Tool:
    return Tool()


@dataclasses.dataclass
class Services:
    tool: RunScoped[Tool]
    label: str


def _services() -> Services:
    return Services(tool=RunScoped(_tool_for), label="x")


def describe_services():
    def when_the_services_are_a_typed_object():
        def it_injects_each_field_by_name():
            seen: list[object] = []

            @on(Asked)
            def show(event: Asked, tool: Tool, label: str) -> None:
                seen.append((type(tool), label))

            EventGraph([show], services=_services()).invoke(Asked())

            assert seen == [(Tool, "x")]

        def when_a_reader_names_a_subclass():
            def it_raises_naming_the_declared_type():
                @on(Asked)
                def show(event: Asked, tool: SubTool) -> None:
                    pass

                with pytest.raises(
                    TypeError,
                    match=r"declares parameter 'tool' as SubTool, but service "
                    r"'tool' gives Tool",
                ):
                    EventGraph([show], services=_services())

        def when_a_reader_permits_none():
            def it_raises_because_a_service_is_never_unset():
                @on(Asked)
                def show(event: Asked, label: str | None) -> None:
                    pass

                with pytest.raises(TypeError, match=r"service 'label' gives str"):
                    EventGraph([show], services=_services())

        def when_the_factory_returns_another_type():
            def it_raises_naming_the_field_type():
                def other_for(config: RunnableConfig) -> SubTool:
                    return SubTool()

                with pytest.raises(
                    TypeError,
                    match=r"Service 'tool' is declared RunScoped\[Tool\], but its "
                    r"factory returns SubTool",
                ):
                    EventGraph(
                        [_shown()],
                        services=Services(tool=RunScoped(other_for), label="x"),
                    )

    def when_the_services_are_a_dict():
        def it_raises_naming_the_fix():
            with pytest.raises(
                TypeError,
                match=r"services= takes an object whose class declares one typed "
                r"field per service, such as a dataclass instance\. Got a dict",
            ):
                EventGraph([_shown()], services={"label": "x"})

    def when_the_services_are_a_list():
        def it_raises_naming_the_fix():
            with pytest.raises(
                TypeError, match=r"services= takes an object .* Got a list"
            ):
                EventGraph([_shown()], services=[Tool()])


def describe_an_injected_reflection():
    def it_reads_the_namespace_model_of_the_graph():
        seen: list[object] = []

        @on(Asked)
        def show(event: Asked, reflection: Reflection) -> None:
            seen.append(reflection._model.run_scoped)

        graph = EventGraph([show], services=_services())
        graph.invoke(Asked())

        assert seen == [graph.namespaces().run_scoped]


def _summary_of(value: Any | None) -> str:
    return str(value)


def describe_a_derived_value():
    def when_a_reader_names_its_return_annotation():
        def it_fits():
            summary = Derived(name="summary", fn=_summary_of)

            @on(Asked)
            def show(event: Asked, summary: str) -> None:
                pass

            scalar = ScalarReducer[Set, Any](name="value", event_type=Set, fn=_value_of)
            EventGraph([show], reducers=[scalar, summary])

    def when_a_reader_names_another_type():
        def it_raises():
            summary = Derived(name="summary", fn=_summary_of)

            @on(Asked)
            def show(event: Asked, summary: object) -> None:
                pass

            scalar = ScalarReducer[Set, Any](name="value", event_type=Set, fn=_value_of)
            with pytest.raises(TypeError, match=r"Derived 'summary' gives str"):
                EventGraph([show], reducers=[scalar, summary])


_WRONG_TYPE = """
from langgraph_events import IntegrationEvent, ScalarReducer


class Set(IntegrationEvent):
    word: str = ""


word = ScalarReducer[Set, int](name="word", event_type=Set, fn=lambda e: e.word)
"""


def describe_the_type_check_of_mypy():
    def when_a_lambda_returns_another_type_than_the_declared_one():
        def it_reports_the_error(tmp_path: Any):
            from mypy import api

            source = tmp_path / "wrong_type.py"
            source.write_text(_WRONG_TYPE)

            stdout, _, status = api.run(["--strict", "--no-incremental", str(source)])

            assert status == 1
            assert (
                'Incompatible return value type (got "str", expected "int | SkipType")'
                in (stdout)
            )
