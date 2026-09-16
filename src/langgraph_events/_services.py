"""Service markers for ``EventGraph(services=...)``."""

from __future__ import annotations

import inspect
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

T = TypeVar("T")


@dataclass(frozen=True)
class RunScoped(Generic[T]):
    """A service derived per run from the node's ``RunnableConfig``.

    Place a ``RunScoped`` value in the name-keyed ``services=`` mapping. The
    factory is called with the run config once per node call. Its result is
    injected under the handler's parameter name.

    The factory must be a plain callable. A coroutine function is rejected,
    because its result would be injected without an ``await``.

    Example::

        services = {"model": RunScoped(lambda config: model_for(config))}

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
        if inspect.iscoroutinefunction(self.factory):
            raise TypeError(
                "RunScoped(factory=...) must not be a coroutine function. The "
                "factory is called synchronously at injection, so its result "
                "would be an un-awaited coroutine. Derive the value with a "
                "plain function."
            )
