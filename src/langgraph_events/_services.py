"""Service markers for ``EventGraph(services=...)``."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

T = TypeVar("T")


def _annotation_source(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Return the object that carries the factory's return annotation.

    A ``functools.partial`` is read through to the wrapped function. A class
    is returned as is. Its instances are the provided type. A callable object
    is read through its ``__call__``.
    """
    while isinstance(factory, functools.partial):
        factory = factory.func
    if isinstance(factory, type) or inspect.isroutine(factory):
        return factory
    return type(factory).__call__


def _missing_return_annotation(factory: Any, source: Callable[..., Any]) -> str:
    """Build the error for a factory whose return annotation is absent."""
    lead = "RunScoped(factory=...) requires a return annotation on the factory"
    why = (
        "The framework compares the factory's return annotation with the "
        "handler parameter's annotation at graph build."
    )
    if getattr(source, "__name__", "") == "<lambda>":
        return (
            f"{lead}, but a lambda cannot carry one. {why} Use a def with a "
            f"return annotation, for example: "
            f"def model_for(config: RunnableConfig) -> ConversationModel."
        )
    if inspect.isroutine(factory) or isinstance(factory, functools.partial):
        label = source.__qualname__
        example = f"def {source.__name__}(config: RunnableConfig) -> ConversationModel"
    else:
        label = f"{type(factory).__qualname__}.__call__"
        example = "def __call__(self, config: RunnableConfig) -> ConversationModel"
    return f"{lead}, but {label!r} has none. {why} Add one, for example: {example}."


@dataclass(frozen=True)
class RunScoped(Generic[T]):
    """A service derived per run from the node's ``RunnableConfig``.

    Place a ``RunScoped`` value in the name-keyed ``services=`` mapping. The
    factory is called with the run config once per node call. Its result is
    injected under the handler's parameter name.

    The factory must be a plain callable. A coroutine function is rejected,
    because its result would be injected without an ``await``.

    Example::

        services = {"model": RunScoped(model_for)}

        @on(SomeEvent)
        def handle(event: SomeEvent, model: ConversationModel) -> None: ...
    """

    factory: Callable[[RunnableConfig], T]

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError(
                f"RunScoped(factory=...) must be callable, got "
                f"{type(self.factory).__name__!r}."
            )
        # A callable object with ``async def __call__`` is not a coroutine
        # function itself, so check its ``__call__`` as well.
        call = getattr(self.factory, "__call__", None)  # noqa: B004
        if inspect.iscoroutinefunction(self.factory) or inspect.iscoroutinefunction(
            call
        ):
            raise TypeError(
                "RunScoped(factory=...) must not be a coroutine function. The "
                "factory is called synchronously at injection, so its result "
                "would be an un-awaited coroutine. Derive the value with a "
                "plain function."
            )
        source = _annotation_source(self.factory)
        if isinstance(source, type):
            return
        if "return" not in getattr(source, "__annotations__", {}):
            raise TypeError(_missing_return_annotation(self.factory, source))
