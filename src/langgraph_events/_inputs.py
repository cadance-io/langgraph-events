"""The inputs of a handler or ``RunScoped`` factory parameter, and their values.

One model serves both: a handler parameter and a factory parameter bind by
the same rules, and one evaluator gives their values in a node call.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, assert_never

from langgraph_events._event_log import EventLog
from langgraph_events._hints import _annotation_accepts_none, _hint_fits, _hint_label
from langgraph_events._reducer import BaseReducer, ReducerNotSetError, StateRead
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from langchain_core.runnables import RunnableConfig

    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reducer import Readable


@dataclass(frozen=True)
class ConfigInput:
    """A parameter annotated ``RunnableConfig``: the run config."""


@dataclass(frozen=True)
class ContextInput:
    """A parameter annotated with the ``context_schema``: the run context."""


@dataclass(frozen=True)
class StoreInput:
    """A parameter annotated ``BaseStore``: the store of the graph."""


@dataclass(frozen=True)
class LogInput:
    """A parameter annotated ``EventLog``: the log at node start."""


@dataclass(frozen=True)
class ReflectionInput:
    """A parameter annotated ``Reflection``: the log with the namespace model."""


@dataclass(frozen=True)
class ReadableInput:
    """A parameter that names a stored reducer or a ``Derived``.

    *required* is ``True`` when the annotation rejects ``None``.
    """

    readable: Readable
    required: bool


@dataclass(frozen=True)
class ValueInput:
    """A parameter that a plain service fills."""

    value: object


@dataclass(frozen=True)
class BoundFactory:
    """The ``RunScoped`` factory of service *name* in one graph, with each
    parameter bound to its input.

    The first *positional* parameters are positional-only. The node call
    passes them by position, and the others by keyword.
    """

    name: str
    factory: Callable[..., object]
    label: str
    params: tuple[Param, ...]
    positional: int = 0

    def call(self, values: Sequence[object]) -> object:
        """Call the factory with *values*, one for each parameter, in order."""
        names = [p.name for p in self.params[self.positional :]]
        keywords = dict(zip(names, values[self.positional :], strict=True))
        return self.factory(*values[: self.positional], **keywords)


ParamInput: TypeAlias = (
    "ConfigInput | ContextInput | StoreInput | LogInput | ReflectionInput"
    " | ReadableInput | ValueInput | BoundFactory"
)

InputKind: TypeAlias = Literal[
    "config",
    "context",
    "store",
    "log",
    "reflection",
    "reducer",
    "derived",
    "service",
    "run_scoped",
]
"""The kind of an input, as the structure views show it."""


@dataclass(frozen=True)
class Param:
    """One parameter of a handler or a factory, bound to its input.

    *default* is the value of the parameter when its input is unset, or
    ``inspect.Parameter.empty`` when the parameter has no default.
    """

    name: str
    source: ParamInput
    default: object = inspect.Parameter.empty

    @property
    def has_default(self) -> bool:
        return self.default is not inspect.Parameter.empty

    @property
    def kind(self) -> InputKind:  # noqa: PLR0911 - one return per kind of input
        """The kind of the input of the parameter."""
        match self.source:
            case ConfigInput():
                return "config"
            case ContextInput():
                return "context"
            case StoreInput():
                return "store"
            case LogInput():
                return "log"
            case ReflectionInput():
                return "reflection"
            case ReadableInput(readable=readable):
                return "reducer" if isinstance(readable, BaseReducer) else "derived"
            case ValueInput():
                return "service"
            case BoundFactory():
                return "run_scoped"
            case _:
                assert_never(self.source)


def classify(
    owner: str,
    param: str,
    hint: object,
    readables: Mapping[str, Readable],
    context_schema: type | None,
) -> ParamInput | None:
    """The input of parameter *param* of *owner*, by annotation then by name.

    An annotation that is exactly ``RunnableConfig``, the *context_schema*,
    ``BaseStore``, ``EventLog`` or ``Reflection`` names a framework input.
    Otherwise the name of a stored reducer or a ``Derived`` names it, and
    the annotation must fit its value. ``None`` means that only a service
    can fill the parameter. *hint* is ``inspect.Parameter.empty`` for a
    parameter with no resolved annotation.
    """
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415
    from langgraph.store.base import BaseStore  # noqa: PLC0415

    from langgraph_events._reflection import Reflection  # noqa: PLC0415

    by_annotation: dict[object, ParamInput] = {
        RunnableConfig: ConfigInput(),
        BaseStore: StoreInput(),
        EventLog: LogInput(),
        Reflection: ReflectionInput(),
    }
    if context_schema is not None:
        by_annotation[context_schema] = ContextInput()
    try:
        framework = by_annotation.get(hint)
    except TypeError:
        framework = None
    if framework is not None:
        return framework
    readable = readables.get(param)
    if readable is None:
        return None
    if hint is inspect.Parameter.empty:
        return ReadableInput(readable, required=False)
    provided = readable._value_type()
    what = f"{readable._describe()} gives {_hint_label(provided)}"
    check_fit(owner, param, hint, _hint_fits(provided, _or_none(hint)), provided, what)
    return ReadableInput(readable, required=not _annotation_accepts_none(hint))


def _or_none(hint: object) -> object:
    """*hint*, widened to accept ``None``. A ``None`` value of a required
    input raises ``ReducerNotSetError`` at run time instead."""
    return typing.Optional[hint]  # noqa: UP045 - a runtime value, not an annotation


def check_fit(
    owner: str,
    param: str,
    declared: object,
    verdict: bool | str,
    provided: object,
    what: str,
    fix: str | None = None,
) -> None:
    """Raise ``TypeError`` for a ``False`` *verdict*. Warn for a reason.

    *what* states the source of the parameter and *provided*, its type, for
    example ``"reducer 'choices' gives list[str]"``. *fix* replaces the
    default fix, which changes the annotation to *provided*.
    """
    wanted = _hint_label(declared)
    if verdict is False:
        fix = fix or (
            f"Change the annotation to {_hint_label(provided)}, or read another input."
        )
        raise TypeError(
            f"{owner} declares parameter {param!r} as {wanted}, but {what}. {fix}"
        )
    if isinstance(verdict, str):
        warn_user(
            f"{owner} declares parameter {param!r} as {wanted}, which Python "
            f"cannot test at run time ({verdict}). The input is not checked "
            f"against it."
        )


class NodeCall:
    """The values of the parameters in one node call of one handler.

    The call reads the state once. Each ``RunScoped`` factory runs at most
    once. Its result serves every reader in the call: the handler, each
    event that the handler handles in the call, each retry attempt, and each
    factory that reads it.
    """

    def __init__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig | None,
        *,
        readables: Mapping[str, Readable],
        model_provider: Callable[[], NamespaceModel],
    ) -> None:
        self._read = StateRead(state)
        self._config = config
        self._readables = readables
        self._model_provider = model_provider
        self._built: dict[str, object] = {}

    def values(self, params: Sequence[Param], reader: str) -> dict[str, object]:
        """The value of each of *params*, by name. *reader* names the handler
        that declares the parameters, for an error."""
        return {p.name: self.value(p, reader, "this handler") for p in params}

    def value(self, param: Param, reader: str, noun: str) -> object:  # noqa: PLR0911 - one return per kind of input
        """The value of *param*. An unset input gives the default of
        *param*. Without a default, it raises an error that names *reader*.
        *noun* is the kind of reader, such as ``"this handler"``.
        """
        source = param.source
        match source:
            case LogInput():
                return self._read.log
            case ReflectionInput():
                from langgraph_events._reflection import Reflection  # noqa: PLC0415

                return Reflection(
                    self._read.log,
                    model=self._model_provider(),
                    reducers=self._readables,
                )
            case ValueInput(value=value):
                return value
            case BoundFactory():
                return self._build(source, reader)
            case ReadableInput(readable=readable, required=required):
                value = readable._read(self._read)
                if value is None and param.has_default:
                    return param.default
                if value is None and required:
                    raise _reducer_not_set(reader, noun, param.name, readable)
                return value
            case ConfigInput() | StoreInput() | ContextInput():
                found = self._framework_value(source)
                if found is not None:
                    return found
                if param.has_default:
                    return param.default
                thing, why, fix = _MISSING[param.kind]
                raise ValueError(
                    f"{reader} needs the {thing} for parameter {param.name!r}, "
                    f"but {why}. {fix}"
                )
            case _:
                assert_never(source)

    def _framework_value(
        self, source: ConfigInput | StoreInput | ContextInput
    ) -> object | None:
        """The run config, the store or the run context, or ``None``."""
        if isinstance(source, ConfigInput):
            return self._config
        from langgraph.runtime import get_runtime  # noqa: PLC0415

        if isinstance(source, ContextInput):
            return get_runtime().context
        if self._config is None:
            return None
        runtime = self._config.get("configurable", {}).get("__pregel_runtime")
        return runtime.store if runtime is not None else None

    def _build(self, factory: BoundFactory, reader: str) -> object:
        """The result of *factory* in this call. *reader* names the reader of
        the result, for the note of an error."""
        if factory.name in self._built:
            return self._built[factory.name]
        values = [self.value(p, factory.label, "this factory") for p in factory.params]
        try:
            result = factory.call(values)
        except Exception as exc:
            # Keep the type, so a caller that catches it still can.
            exc.add_note(
                f"{reader} requested the run-scoped service {factory.name!r}, "
                f"but its factory raised."
            )
            raise
        self._built[factory.name] = result
        return result


_MISSING = {
    "config": (
        "run config",
        "runtime config is missing",
        "Run the graph through an EventGraph run method.",
    ),
    "store": (
        "store",
        "no store is configured",
        "Pass store=... to EventGraph(...).",
    ),
    "context": (
        "run context",
        "the run has no context",
        "Pass context=... to the run call.",
    ),
}
"""For each framework input that a run can lack: what it is, why it is
missing, and the fix."""


def _reducer_not_set(
    reader: str, noun: str, param: str, readable: Readable
) -> ReducerNotSetError:
    """A ``ReducerNotSetError`` with the fix. *reader* names the handler or
    the ``RunScoped`` factory that reads *readable*. *noun* is its kind."""
    return ReducerNotSetError(
        f"{reader} requires {readable._describe()} to be set, but its value "
        f"is None. {readable._not_set_hint(noun)}give the parameter a "
        f"default, or change the annotation to permit None (e.g. "
        f"'{param}: <type> | None')."
    )
