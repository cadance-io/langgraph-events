"""The inputs of a handler or ``RunScoped`` factory parameter, and their values.

One model serves both: a handler parameter and a factory parameter bind by
the same rules, and one evaluator gives their values in a node call.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    assert_never,
)

from langgraph_events._event_log import EventLog
from langgraph_events._hints import (
    Misfit,
    Untestable,
    Verdict,
    _annotation_accepts_none,
    _hint_fits,
    _hint_label,
    _resolve_parameter_hints,
)
from langgraph_events._reducer import BaseReducer, ReducerNotSetError, StateRead
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from langchain_core.runnables import RunnableConfig

    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reducer import Derived, Readable

T = TypeVar("T")


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
    readable_of: Callable[[str], Readable | None],
    context_schema: type | None,
) -> ParamInput | None:
    """The input of parameter *param* of *owner*, by annotation then by name.

    An annotation that is exactly ``RunnableConfig``, the *context_schema*,
    ``BaseStore``, ``EventLog`` or ``Reflection`` names a framework input.
    Otherwise the name of a stored reducer or a ``Derived`` names it, and
    the annotation must fit its value. *readable_of* gives the stored
    reducer or the ``Derived`` of a name, or ``None``. ``None`` means that
    only a service can fill the parameter. *hint* is the resolved
    annotation, see :func:`resolved_hint`.
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
    readable = readable_of(param)
    if readable is None:
        return None
    provided = readable._value_type()
    what = f"{readable._describe()} gives {_hint_label(provided)}"
    check_fit(owner, param, hint, _hint_fits(provided, _or_none(hint)), provided, what)
    return ReadableInput(readable, required=not _annotation_accepts_none(hint))


def _or_none(hint: object) -> object:
    """*hint*, widened to accept ``None``. A ``None`` value of a required
    input raises ``ReducerNotSetError`` at run time instead."""
    return typing.Optional[hint]  # noqa: UP045 - a runtime value, not an annotation


def resolved_hint(
    owner: str, param: str, hints: Mapping[str, Any], errors: Mapping[str, str]
) -> object:
    """The resolved annotation of parameter *param* of *owner*.

    Raises ``TypeError`` when the annotation does not resolve, or when the
    parameter has none. Each message names the fix.
    """
    if param in errors:
        raise TypeError(
            f"{owner}: the annotation of parameter {param!r} does not resolve "
            f"({errors[param]}). Make every name in it importable at run time. "
            f"A usual cause is an import under 'if TYPE_CHECKING:', which the "
            f"ruff rules TC001 to TC003 can move there. Keep the import at run "
            f"time with '# noqa: TC002'."
        )
    if param not in hints:
        raise TypeError(
            f"{owner} has no annotation on parameter {param!r}. Annotate it with "
            f"the type of its input: an exact library type such as "
            f"RunnableConfig or EventLog, or the value type of the reducer, the "
            f"Derived or the service that it names."
        )
    return hints[param]


def check_fit(
    owner: str,
    param: str,
    declared: object,
    verdict: Verdict,
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
    if isinstance(verdict, Misfit):
        fix = fix or (
            f"Change the annotation to {_hint_label(provided)}, or read another input."
        )
        raise TypeError(
            f"{owner} declares parameter {param!r} as {wanted}, but {what}. {fix}"
        )
    if isinstance(verdict, Untestable):
        warn_user(
            f"{owner} declares parameter {param!r} as {wanted}, which Python "
            f"cannot test at run time ({verdict.reason}). The input is not "
            f"checked against it."
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
            case ReadableInput():
                return _readable_value(source, param, self._read, reader, noun)
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


def _readable_value(
    source: ReadableInput, param: Param, read: StateRead, reader: str, noun: str
) -> object:
    """The value of a reducer or a ``Derived`` in *read*. An unset value gives
    the default of *param*, or raises ``ReducerNotSetError`` for a required
    one."""
    value = source.readable._read(read)
    if value is None and param.has_default:
        return param.default
    if value is None and source.required:
        raise _reducer_not_set(reader, noun, param.name, source.readable)
    return value


@dataclass(frozen=True)
class BoundDerived(Generic[T]):
    """A :class:`Derived` of one graph, with each parameter bound to its
    input: a stored reducer, another ``Derived`` or the log.

    One ``Derived`` can serve several graphs, so the binding is not stored
    on it.
    """

    derived: Derived[T]
    params: tuple[Param, ...]

    @property
    def name(self) -> str:
        return self.derived.name

    @property
    def sources(self) -> tuple[BaseReducer, ...]:
        """The stored reducers that the value reads, also through another
        ``Derived``, in input order."""
        found: dict[str, BaseReducer] = {}
        for p in self.params:
            if isinstance(p.source, ReadableInput):
                for reducer in p.source.readable.sources:
                    found.setdefault(reducer.name, reducer)
        return tuple(found.values())

    @property
    def _reads_log(self) -> bool:
        return any(
            isinstance(p.source, LogInput)
            or (isinstance(p.source, ReadableInput) and p.source.readable._reads_log)
            for p in self.params
        )

    def _read(self, read: StateRead) -> T:
        return read.computed(self)

    def _compute(self, read: StateRead) -> T:
        values: dict[str, object] = {}
        for p in self.params:
            match p.source:
                case LogInput():
                    values[p.name] = read.log
                case ReadableInput():
                    values[p.name] = _readable_value(
                        p.source, p, read, self._describe(), "this Derived"
                    )
                case _:
                    raise AssertionError(f"{p.source!r} is no input of a Derived.")
        try:
            return self.derived.fn(**values)
        except Exception as exc:
            exc.add_note(f"Derived {self.name!r} raised while it computed its value.")
            raise

    def _value_type(self) -> object:
        """The return annotation of ``fn``. The bind refuses a ``Derived``
        without one."""
        return _resolve_parameter_hints(self.derived.fn)[0]["return"]

    def _describe(self) -> str:
        return f"Derived {self.name!r}"

    def _not_set_hint(self, reader: str) -> str:
        return f"Make Derived {self.name!r} return a non-None value, or "


def bind_derived(
    reducers: Mapping[str, BaseReducer],
    derived: Mapping[str, Derived[Any]],
    context_schema: type | None = None,
) -> dict[str, BoundDerived[Any]]:
    """Bind each ``Derived`` of *derived* to its inputs, by the rule of a
    handler parameter.

    A parameter annotated ``EventLog`` reads the log. Each other parameter
    names a stored reducer or another ``Derived``, and its annotation must
    fit that value. A ``Derived`` that another one reads is bound once.
    Raises ``TypeError`` for a parameter that has no annotation, one that
    does not resolve, one that names no input, and one that names a run
    input such as the config. Raises ``TypeError`` that names the cycle when
    ``Derived`` values read each other in a cycle.
    """
    bound: dict[str, BoundDerived[Any]] = {}

    def bind(name: str, path: tuple[str, ...]) -> BoundDerived[Any]:
        if name in path:
            cycle = " -> ".join((*path[path.index(name) :], name))
            raise TypeError(
                f"Derived values read each other in a cycle: {cycle}. A "
                f"Derived must not read itself, directly or through another "
                f"Derived. Remove one read from the cycle."
            )
        if name in bound:
            return bound[name]

        def readable_of(dep: str) -> Readable | None:
            if dep in derived:
                return bind(dep, (*path, name))
            return reducers.get(dep)

        value = derived[name]
        owner = f"Derived {name!r}"
        hints, errors = _resolve_parameter_hints(value.fn)
        params = tuple(
            _derived_param(
                owner, param, hints, errors, readable_of, context_schema, no_input
            )
            for param in value._parameters()
        )
        _require_return_annotation(owner, hints, errors)
        bound[name] = BoundDerived(value, params)
        return bound[name]

    def no_input(owner: str, param: inspect.Parameter) -> str:
        default_hint = (
            " A default does not make a parameter optional. Bind a constant "
            "in a closure."
            if param.default is not param.empty
            else ""
        )
        return (
            f"{owner} reads {param.name!r}, which names no reducer and no "
            f"Derived.{default_hint} Available reducers: {sorted(reducers)}. "
            f"Available Derived values: {sorted(derived)}. To read the "
            f"EventLog, annotate the parameter with EventLog."
        )

    for name in derived:
        bind(name, ())
    return bound


def _derived_param(
    owner: str,
    param: inspect.Parameter,
    hints: Mapping[str, Any],
    errors: Mapping[str, str],
    readable_of: Callable[[str], Readable | None],
    context_schema: type | None,
    no_input: Callable[[str, inspect.Parameter], str],
) -> Param:
    """Bind one parameter of a ``Derived``, or raise ``TypeError``."""
    if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
        raise TypeError(
            f"{owner} has the variadic or positional-only parameter "
            f"{param.name!r}. Each parameter must name one input."
        )
    hint = resolved_hint(owner, param.name, hints, errors)
    source = classify(owner, param.name, hint, readable_of, context_schema)
    if source is None:
        raise TypeError(no_input(owner, param))
    bound = Param(param.name, source, param.default)
    if not isinstance(source, LogInput | ReadableInput):
        raise TypeError(
            f"{owner} reads the {bound.kind} in parameter {param.name!r}. A "
            f"Derived reads only stored reducers, Derived values and the "
            f"EventLog, because a stream frame and a reflection have no run. "
            f"Read the value in a handler or in a RunScoped factory."
        )
    if isinstance(source, LogInput) and bound.has_default:
        raise TypeError(
            f"{owner} gives parameter {param.name!r} a default. A default does "
            f"not make a parameter optional, and the EventLog is always set. "
            f"Remove the default, or bind a constant in a closure."
        )
    return bound


def _require_return_annotation(
    owner: str, hints: Mapping[str, Any], errors: Mapping[str, str]
) -> None:
    """Raise ``TypeError`` for a ``Derived`` whose return annotation is
    missing or does not resolve."""
    if "return" in errors:
        raise TypeError(
            f"{owner} has a return annotation that does not resolve "
            f"({errors['return']}). Make every name in it importable at run "
            f"time."
        )
    if "return" not in hints:
        raise TypeError(
            f"{owner} has no return annotation. The framework checks the "
            f"annotation of each reader against it. A lambda cannot have one, "
            f"so use a def, for example: def stage_of(placed: bool | None) -> "
            f"str."
        )


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
