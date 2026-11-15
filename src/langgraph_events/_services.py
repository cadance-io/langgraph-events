"""Service markers for ``EventGraph(services=...)``."""

from __future__ import annotations

import functools
import inspect
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

from langgraph_events._handler import (
    BoundHandler,
    ServiceCandidate,
    _annotation_accepts_none,
    _annotation_source,
    _context_hint,
    _resolve_hints_and_errors,
    _resolve_parameter_hints,
)
from langgraph_events._reducer import BaseReducer
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langgraph_events._handler import HandlerMeta
    from langgraph_events._reducer import Readable

T = TypeVar("T")


def _missing_return_annotation(
    factory: Callable[..., object], source: Callable[..., Any]
) -> str:
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
    """A service that a factory builds once per node call.

    Place a ``RunScoped`` value in the name-keyed ``services=`` mapping. Its
    result is injected under the handler's parameter name.

    Each parameter of the factory binds like a handler parameter. A
    parameter annotated ``RunnableConfig`` gets the run config. A parameter
    annotated with the graph's ``context_schema`` gets the run context. By
    name, ``log`` gets the ``EventLog``, and any other parameter gets the
    stored reducer, the ``Derived`` or the service of that name. Each
    parameter must have an annotation and no default value.

    The factory runs only for a handler or a factory that declares it,
    once per node call. One result serves every reader in that call.

    The factory must be a plain callable. A coroutine function is rejected,
    because its result would be injected without an ``await``.

    Example::

        def model_for(config: RunnableConfig, language: str) -> ConversationModel:
            ...

        services = {"model": RunScoped(model_for)}

        @on(SomeEvent)
        def handle(event: SomeEvent, model: ConversationModel) -> None: ...
    """

    factory: Callable[..., T]

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

    def _label(self, name: str) -> str:
        """The subject of an error about the factory of service *name*."""
        source = _annotation_source(self.factory)
        return f"RunScoped factory {_factory_label(source)!r} for service {name!r}"

    def _parameters(self, name: str) -> tuple[inspect.Parameter, ...]:
        """The parameters that the node call must fill. A keyword that a
        ``functools.partial`` binds is filled already."""
        bound: set[str] = set()
        factory: object = self.factory
        while isinstance(factory, functools.partial):
            bound.update(factory.keywords)
            factory = factory.func
        try:
            params = inspect.signature(self.factory).parameters.values()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{self._label(name)}: Python cannot read the parameters of "
                f"the factory ({exc}). Each parameter names one input, so "
                f"pass a function, a class or a callable object whose "
                f"signature Python can read."
            ) from exc
        return tuple(p for p in params if p.name not in bound)

    def _hints(self) -> tuple[dict[str, Any], dict[str, str]]:
        """The resolved parameter annotations of the factory, and the errors."""
        return _resolve_parameter_hints(self.factory)

    def provided_type(self, name: str) -> object:
        """Resolve the return annotation of the factory of service *name*.

        The build compares it with the handler parameter's annotation. A
        return annotation that does not resolve raises ``TypeError``.
        """
        source = _annotation_source(self.factory)
        if isinstance(source, type):
            return source
        hints, errors = _resolve_hints_and_errors(source)
        if "return" in errors:
            raise TypeError(
                f"RunScoped factory {_factory_label(source)!r} for service "
                f"{name!r} has a "
                f"return annotation that did not resolve ({errors['return']}). "
                f"The framework compares it with each handler's parameter "
                f"annotation at graph build, so it cannot be left unresolvable. "
                f"Make every name in the annotation importable at run time, or "
                f"declare the class at module level."
            )
        return _resolve_self(hints["return"], source)


@dataclass(frozen=True)
class ConfigInput:
    """A factory parameter annotated ``RunnableConfig``: the run config."""


@dataclass(frozen=True)
class ContextInput:
    """A factory parameter annotated with the ``context_schema``: the run
    context."""


@dataclass(frozen=True)
class LogInput:
    """The factory parameter named ``log``: the ``EventLog`` of the node."""


@dataclass(frozen=True)
class ReadableInput:
    """A factory parameter that names a stored reducer or a ``Derived``.

    *required* is ``True`` when the annotation rejects ``None``. A ``None``
    value then raises ``ReducerNotSetError``.
    """

    readable: Readable
    required: bool


@dataclass(frozen=True)
class ValueInput:
    """A factory parameter that names a plain service."""

    value: object


FactoryInput: TypeAlias = (
    "ConfigInput | ContextInput | LogInput | ReadableInput | ValueInput | BoundFactory"
)


@dataclass(frozen=True)
class BoundFactory:
    """The ``RunScoped`` factory of service *name* in one graph, with each
    parameter bound to its input.

    The first *positional* parameters are positional-only. The node call
    passes them by position, and the others by keyword.
    """

    name: str
    scoped: RunScoped[Any]
    params: tuple[tuple[str, FactoryInput], ...]
    positional: int = 0

    @property
    def label(self) -> str:
        """The subject of an error about this factory."""
        return self.scoped._label(self.name)

    @property
    def inputs(self) -> tuple[str, ...]:
        """The parameter names of the factory, in signature order."""
        return tuple(param for param, _ in self.params)

    @property
    def reads(self) -> tuple[str, ...]:
        """The parameter names that bind by name to a stored reducer, a
        ``Derived`` or another service."""
        return tuple(
            param
            for param, source in self.params
            if isinstance(source, ReadableInput | ValueInput | BoundFactory)
        )

    def call(self, values: Sequence[object]) -> object:
        """Call the factory with *values*, one for each parameter, in order."""
        names = self.inputs[self.positional :]
        keywords = dict(zip(names, values[self.positional :], strict=True))
        return self.scoped.factory(*values[: self.positional], **keywords)


class _FactoryBinder:
    """Bind each ``RunScoped`` factory of a name-keyed ``services=`` to its
    inputs, for one graph."""

    def __init__(
        self,
        services: Mapping[str, Any],
        provided: Mapping[str, object],
        readables: Mapping[str, Readable],
        context_schema: type | None,
    ) -> None:
        self._services = services
        self._provided = provided
        self._readables = readables
        self._context_schema = context_schema
        self._bound: dict[str, BoundFactory] = {}

    def bind_all(self) -> dict[str, BoundFactory]:
        for name, value in self._services.items():
            if isinstance(value, RunScoped):
                self._bind(name, ())
        return self._bound

    def _bind(self, name: str, path: tuple[str, ...]) -> BoundFactory:
        if name in path:
            cycle = " -> ".join((*path[path.index(name) :], name))
            raise TypeError(
                f"RunScoped factories read each other in a cycle: {cycle}. A "
                f"factory must not read itself, directly or through another "
                f"factory. Remove one read from the cycle."
            )
        if name in self._bound:
            return self._bound[name]
        scoped: RunScoped[Any] = self._services[name]
        hints, errors = scoped._hints()
        params: list[tuple[str, FactoryInput]] = []
        positional = 0
        for param in scoped._parameters(name):
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional += 1
            source = self._bind_param(name, scoped, param, hints, errors, path)
            params.append((param.name, source))
        bound = BoundFactory(name, scoped, tuple(params), positional)
        self._bound[name] = bound
        return bound

    def _bind_param(
        self,
        name: str,
        scoped: RunScoped[Any],
        param: inspect.Parameter,
        hints: Mapping[str, Any],
        errors: Mapping[str, str],
        path: tuple[str, ...],
    ) -> FactoryInput:
        from langchain_core.runnables import RunnableConfig  # noqa: PLC0415

        label = scoped._label(name)
        p = param.name
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{label} has the variadic parameter {p!r}. Each parameter "
                f"must name one input. Replace it with one parameter per input."
            )
        if param.default is not inspect.Parameter.empty:
            raise TypeError(
                f"{label} gives parameter {p!r} a default value. A default "
                f"hides a missing input. Remove the default."
            )
        if p in errors:
            raise TypeError(
                f"{label} has an annotation on parameter {p!r} that did not "
                f"resolve ({errors[p]}). Make every name in the annotation "
                f"importable at run time."
            )
        if p not in hints:
            raise TypeError(
                f"{label} has no annotation on parameter {p!r}. Annotate it "
                f"with RunnableConfig, the context schema, EventLog, or the "
                f"type of the reducer, Derived or service that it names."
            )
        declared = hints[p]
        if declared is RunnableConfig:
            return ConfigInput()
        if self._context_schema is not None and declared is self._context_schema:
            return ContextInput()
        if p == "log":
            return self._log_input(label, declared)
        readable = self._readables.get(p)
        if readable is not None:
            self._check(
                label, p, declared, readable._fits(declared), readable._describe_value()
            )
            return ReadableInput(readable, not _annotation_accepts_none(declared))
        if p in self._services:
            return self._service_input(label, p, declared, (*path, name))
        reducers = sorted(
            n for n, r in self._readables.items() if isinstance(r, BaseReducer)
        )
        derived = sorted(self._readables.keys() - set(reducers))
        context = _context_hint([ServiceCandidate(p, declared)], self._context_schema)
        raise TypeError(
            f"{label} reads parameter {p!r}, which names no reducer, no Derived "
            f"and no service. Rename the parameter to one of the available "
            f"names: reducers {reducers}, Derived values {derived}, services "
            f"{sorted(self._services)}, or 'log' for the EventLog.{context}"
        )

    def _log_input(self, label: str, declared: object) -> LogInput:
        from beartype.door import is_subhint  # noqa: PLC0415

        from langgraph_events._event_log import EventLog  # noqa: PLC0415

        if "log" in self._services:
            raise TypeError(
                f"{label} reads 'log', and a service has the name 'log'. In a "
                f"factory, 'log' names the EventLog. Rename the service."
            )
        self._check(
            label,
            "log",
            declared,
            _satisfies(is_subhint, EventLog, declared),
            "'log' is the EventLog",
            fix="EventLog",
        )
        return LogInput()

    def _service_input(
        self, label: str, p: str, declared: object, path: tuple[str, ...]
    ) -> FactoryInput:
        from beartype.door import is_bearable  # noqa: PLC0415

        value = self._services[p]
        if isinstance(value, RunScoped):
            provided = self._provided[p]
            returns = _hint_label(provided)
            self._check(
                label,
                p,
                declared,
                _hint_fits(provided, declared),
                f"service {p!r} is RunScoped("
                f"{_factory_label(_annotation_source(value.factory))}), which "
                f"returns {returns}",
                fix=returns,
            )
            return self._bind(p, path)
        self._check(
            label,
            p,
            declared,
            _satisfies(is_bearable, value, declared),
            f"service {p!r} is {type(value).__qualname__}",
            fix=type(value).__qualname__,
        )
        return ValueInput(value)

    @staticmethod
    def _check(
        label: str,
        p: str,
        declared: object,
        verdict: bool | str,
        what: str,
        *,
        fix: str | None = None,
    ) -> None:
        """Raise ``TypeError`` for a ``False`` verdict. Warn for a reason."""
        wanted = _hint_label(declared)
        if verdict is False:
            change = (
                f"Change the annotation to {fix}"
                if fix is not None
                else "Change the annotation"
            )
            raise TypeError(
                f"{label} declares parameter {p!r} as {wanted}, but {what}. "
                f"{change}, or read another input."
            )
        if isinstance(verdict, str):
            warn_user(
                f"{label} declares parameter {p!r} as {wanted}, which Python "
                f"cannot test at run time ({verdict}). The input is not "
                f"checked against it."
            )


class TypedServices:
    """The sequence form of ``services=``: each service fills the handler
    parameters that its type satisfies. Two services of one type are refused.
    """

    def __init__(self, services: Sequence[Any]) -> None:
        self._by_type: dict[type, Any] = {}
        for s in services:
            if isinstance(s, RunScoped):
                raise TypeError(
                    "RunScoped is only supported in the name-keyed mapping form: "
                    "services={'name': RunScoped(factory)}. The type-keyed sequence "
                    "form resolves by annotation, and no handler annotates RunScoped."
                )
            t = type(s)
            if t in self._by_type:
                raise TypeError(
                    f"EventGraph(services=...) has two instances of type "
                    f"{t.__name__!r} — same-type collision. Register only "
                    f"one instance, switch to the name-keyed mapping form "
                    f"(services={{'primary': a, 'backup': b}}), or use a "
                    f"subclass to distinguish them."
                )
            self._by_type[t] = s

    def received(self) -> str:
        """State the types that ``services=`` holds, for an error."""
        if not self._by_type:
            return "services= has no service."
        names = sorted(t.__qualname__ for t in self._by_type)
        return f"services= has the types {names}."

    def bind_factories(
        self, readables: Mapping[str, Readable], context_schema: type | None
    ) -> Mapping[str, BoundFactory]:
        """The type-keyed form holds no ``RunScoped`` factory."""
        return {}

    def bind(
        self,
        meta: HandlerMeta,
        context_schema: type | None,
        factories: Mapping[str, BoundFactory],
    ) -> BoundHandler:
        """Fill each candidate of *meta* whose annotation is a class with the
        service of that class.

        Raises ``TypeError`` for a service that is an instance of
        *context_schema*: the run context fills a parameter of that type.

        The service is the unique registered one whose type is a subclass of
        (or equal to) the annotation. Multi-match is rejected — the user must
        disambiguate by narrowing the annotation or registering only one
        matching service.
        """
        # A ``TypedDict`` refuses ``issubclass``, and its instance is a dict.
        if context_schema is not None and not typing.is_typeddict(context_schema):
            for t in self._by_type:
                if issubclass(t, context_schema):
                    raise TypeError(
                        f"services= holds an instance of {t.__qualname__}, the "
                        f"context_schema of the graph. The run context fills a "
                        f"parameter of that type, so the service is never "
                        f"injected. Pass the value with context= on each run "
                        f"call, and remove it from services=."
                    )
        detected: list[tuple[str, object]] = []
        for candidate in meta.service_candidates:
            hint = candidate.annotation
            if not isinstance(hint, type):
                continue
            # ``object`` matches every registered type; treating it as a service
            # claim would silently consume an unrelated instance. Skip — the
            # caller meant "untyped" and ``BoundHandler.require_claimed`` will
            # surface the missing-source error at graph build.
            if hint is object:
                continue
            # Exact-type match wins over subclass-only match — it lets users
            # register a base + subclass pair (``[BaseChatModel(), Anthropic()]``)
            # and resolve `param: BaseChatModel` to the base instance unambiguously.
            if hint in self._by_type:
                detected.append((candidate.param, self._by_type[hint]))
                continue
            matches = [t for t in self._by_type if issubclass(t, hint)]
            if len(matches) > 1:
                names = ", ".join(sorted(t.__name__ for t in matches))
                raise TypeError(
                    f"{meta.label} parameter {candidate.param!r} "
                    f"annotated as {hint.__name__!r} matches multiple "
                    f"registered services ({names}). Disambiguate by "
                    f"narrowing the annotation, or register only one "
                    f"service that satisfies it."
                )
            if matches:
                detected.append((candidate.param, self._by_type[matches[0]]))
        return BoundHandler(meta, tuple(detected))


class NamedServices:
    """The mapping form of ``services=``: each service fills the handler
    parameter of its name, so two instances of one type can coexist (a
    primary and a backup chat model). Annotation-free parameters are eligible.
    """

    def __init__(self, services: Mapping[str, Any]) -> None:
        self._by_name = dict(services)
        # The type that each ``RunScoped`` factory returns. Resolved here, so
        # a bad return annotation does not wait for the first handler.
        self._run_scoped_types = {
            name: value.provided_type(name)
            for name, value in self._by_name.items()
            if isinstance(value, RunScoped)
        }

    def received(self) -> str:
        """State the name keys that ``services=`` holds, for an error."""
        if not self._by_name:
            return "services= has no service."
        return f"services= has the name keys {sorted(self._by_name)}."

    def bind_factories(
        self, readables: Mapping[str, Readable], context_schema: type | None
    ) -> Mapping[str, BoundFactory]:
        """Bind each ``RunScoped`` factory to its inputs, by the handler rules.

        Raises ``TypeError`` for a service whose name a reducer or a
        ``Derived`` also has, because a parameter cannot name both. Raises
        ``TypeError`` for each factory parameter that cannot bind, and for a
        cycle among factories.
        """
        for name in self._by_name:
            readable = readables.get(name)
            if readable is not None:
                kind = "reducer" if isinstance(readable, BaseReducer) else "Derived"
                raise TypeError(
                    f"services= has the name key {name!r}, and a {kind} has the "
                    f"same name. A parameter cannot name both. Rename the "
                    f"service or the {kind}."
                )
        return _FactoryBinder(
            self._by_name, self._run_scoped_types, readables, context_schema
        ).bind_all()

    def bind(
        self,
        meta: HandlerMeta,
        context_schema: type | None,
        factories: Mapping[str, BoundFactory],
    ) -> BoundHandler:
        """Fill each candidate of *meta* with the service of its name, then
        check each service against the annotation. A ``RunScoped`` service
        fills its candidate with its bound factory from *factories*."""
        services = tuple(
            (c.param, factories.get(c.param, self._by_name[c.param]))
            for c in meta.service_candidates
            if c.param in self._by_name
        )
        bound = BoundHandler(meta, services)
        self._verify(bound)
        return bound

    def _verify(self, bound: BoundHandler) -> None:
        """Raise if a service does not satisfy the handler annotation.

        Name-keyed binding matches on the parameter name and needs no
        annotation. When the parameter has an annotation that resolved, the
        framework checks the registered service against it. A plain value
        must be an instance of the annotation. A ``RunScoped`` factory must
        return a subtype of the annotation. The comparison uses
        ``beartype.door``. It handles ``Union``, a runtime ``Protocol`` with
        methods only, and a generic alias. A ``NewType`` is nominal for a
        factory and erased for a plain value.
        """
        bound_params = {param for param, _ in bound.services}
        annotated = [
            c
            for c in bound.meta.service_candidates
            if c.param in bound_params and c.has_resolved_annotation
        ]
        if not annotated:
            return
        # A cold import costs about 200 ms. Only the first graph in the process
        # with an annotated name-keyed parameter pays it.
        from beartype.door import is_bearable, is_subhint  # noqa: PLC0415

        handler = bound.meta.label
        for candidate in annotated:
            param, declared = candidate.param, candidate.annotation
            value = self._by_name[param]
            label = _hint_label(declared)
            if isinstance(value, RunScoped):
                provided = self._run_scoped_types[param]
                # ``Any`` and a bare type variable say nothing about the result.
                if provided is Any or isinstance(provided, TypeVar):
                    continue
                verdict = _satisfies(is_subhint, provided, declared)
                if verdict is False:
                    factory = _factory_label(_annotation_source(value.factory))
                    returns = _hint_label(provided)
                    raise TypeError(
                        f"{handler} declares parameter {param!r} as "
                        f"{label}, but services[{param!r}] is RunScoped({factory}) "
                        f"and that factory returns {returns}. Change the "
                        f"annotation to {returns}, or register a factory that "
                        f"returns {label}."
                    )
            else:
                verdict = _satisfies(is_bearable, value, declared)
                if verdict is False:
                    if value is None:
                        what = "None"
                        fix = f"Widen the annotation to {label} | None"
                    else:
                        what = f"an instance of {type(value).__qualname__}"
                        fix = "Widen the annotation"
                    raise TypeError(
                        f"{handler} declares parameter {param!r} as "
                        f"{label}, but services[{param!r}] is {what}, not an "
                        f"instance of {label}. {fix}, or register a value that "
                        f"satisfies it."
                    )
            if isinstance(verdict, str):
                warn_user(
                    f"{handler} declares parameter {param!r} as {label}, "
                    f"which Python cannot test at run time ({verdict}). The "
                    f"service under services[{param!r}] is not checked against it."
                )


Services = TypedServices | NamedServices


def parse_services(services: Sequence[Any] | Mapping[str, Any] | None) -> Services:
    """Parse ``EventGraph(services=...)``. A mapping is name-keyed. A sequence
    is type-keyed."""
    if isinstance(services, Mapping):
        return NamedServices(services)
    return TypedServices(services or ())


def _resolve_self(hint: object, source: Callable[..., object]) -> object:
    """Replace ``typing.Self`` with the class the method is bound to.

    ``get_type_hints`` leaves ``Self`` as is. A method bound to a class
    provides that class. A method bound to an instance provides its type. An
    unbound function has no owner, so the hint is treated as ``Any``.
    """
    if hint is not typing.Self:
        return hint
    owner = getattr(source, "__self__", None)
    if owner is None:
        return Any
    return owner if isinstance(owner, type) else type(owner)


def _hint_label(hint: object) -> str:
    """Render an annotation for an error message.

    A generic alias such as ``list[str]`` forwards ``__qualname__`` to its
    origin, so only a plain class uses it.
    """
    if isinstance(hint, type):
        return hint.__qualname__
    return repr(hint)


def _factory_label(factory: object) -> str:
    """Render a ``RunScoped`` factory for an error message."""
    if inspect.isroutine(factory):
        return factory.__qualname__
    return _hint_label(factory)


def _or_none(hint: object) -> object:
    """*hint*, widened to accept ``None``."""
    return typing.Optional[hint]  # noqa: UP045 - a runtime value, not an annotation


def _erased(hint: object) -> object:
    """*hint* without its type arguments, also inside a union.

    ``list[str]`` becomes ``list``. A hint with no class origin, such as
    ``Annotated``, stays as is.
    """
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return typing.Union[tuple(_erased(a) for a in typing.get_args(hint))]  # noqa: UP007 - a runtime value
    if isinstance(origin, type):
        return origin
    return hint


def _hint_fits(provided: object, declared: object) -> bool | str:
    """Whether *provided* is a subhint of *declared*. ``Any`` and a bare type
    variable say nothing, so they fit. A string is the reason that the check
    cannot run."""
    if provided is Any or isinstance(provided, TypeVar):
        return True
    from beartype.door import is_subhint  # noqa: PLC0415

    return _satisfies(is_subhint, provided, declared)


def _satisfies(
    check: Callable[[Any, Any], bool], subject: object, declared: object
) -> bool | str:
    """Apply a ``beartype.door`` check. A string is the reason it cannot run.

    ``typing`` refuses a ``Protocol`` that is not ``runtime_checkable``, and
    beartype refuses a hint it cannot test at run time, at any nesting depth.
    Such an annotation stays a valid static contract, so the build must not
    fail on it. The caller warns instead.
    """
    from beartype.roar import BeartypeException  # noqa: PLC0415

    try:
        return check(subject, declared)
    except (TypeError, BeartypeException) as exc:
        return f"{type(exc).__name__}: {exc}"
