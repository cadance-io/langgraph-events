"""Service markers for ``EventGraph(services=...)``."""

import annotationlib
import dataclasses
import functools
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langgraph_events._handler import (
    BoundHandler,
    ServiceCandidate,
    refuse_unclaimed,
)
from langgraph_events._hints import (
    UnsettledTypeError,
    _annotation_source,
    _hint_label,
    _normal,
    _open_parameters,
    _resolve_parameter_hints,
    _same_type,
)
from langgraph_events._inputs import (
    BoundFactory,
    Param,
    ValueInput,
    classify,
    require_same_type,
    resolved_hint,
)
from langgraph_events._reducer import BaseReducer

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph_events._handler import HandlerMeta
    from langgraph_events._reducer import Readable


def _missing_return_annotation(
    factory: Callable[..., object], source: Callable[..., Any]
) -> str:
    """Build the error for a factory whose return annotation is absent."""
    lead = "RunScoped(factory=...) requires a return annotation on the factory"
    why = (
        "The framework compares the factory's return annotation with the "
        "annotation of each handler or factory parameter that reads it, at "
        "graph build."
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
class RunScoped[T]:
    """A service that a factory builds once per node call.

    Give the ``services=`` object a field typed ``RunScoped[T]`` that holds
    it. The result has the type ``T``, and is injected under the field name.

    Each parameter of the factory binds by the rules of a handler
    parameter. An annotation of exactly ``RunnableConfig``, the graph's
    ``context_schema``, ``EventLog``, ``BaseStore`` or ``Reflection`` gets
    that input. Each other parameter gets the stored reducer, the
    ``Derived`` or the name-keyed service of its name, and its annotation
    must name its declared type. Each parameter must have an annotation that
    resolves. A default is the value when the input is unset. A
    parameter that names nothing raises ``TypeError`` at graph build.

    The factory runs only for a handler or a factory that declares it,
    once per node call, before the invariants of the handler. One result
    serves every event that the node handles in that call, every retry
    attempt, and every factory that reads it.

    The factory must be a plain callable. A coroutine function is rejected,
    because its result would be injected without an ``await``.

    Example::

        def model_for(config: RunnableConfig, language: str) -> ConversationModel:
            ...

        language = ScalarReducer[LanguageSet, str](
            name="language", event_type=LanguageSet, fn=lambda e: e.language
        )

        @dataclass
        class Services:
            model: RunScoped[ConversationModel]

        services = Services(model=RunScoped(model_for))

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
        annotations = annotationlib.get_annotations(
            source, format=annotationlib.Format.FORWARDREF
        )
        if "return" not in annotations:
            raise TypeError(_missing_return_annotation(self.factory, source))

    def _label(self, name: str) -> str:
        """The subject of an error about the factory of service *name*."""
        source = _annotation_source(self.factory)
        return f"RunScoped factory {_factory_label(source)!r} for service {name!r}"

    def _parameters(self, name: str) -> tuple[inspect.Parameter, ...]:
        """The parameters that the node call must fill. A keyword that a
        ``functools.partial`` binds is filled already."""
        try:
            return _open_parameters(self.factory)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{self._label(name)}: Python cannot read the parameters of "
                f"the factory ({exc}). Each parameter names one input, so "
                f"pass a function, a class or a callable object whose "
                f"signature Python can read."
            ) from exc

    def _hints(self) -> tuple[dict[str, Any], dict[str, str]]:
        """The resolved parameter annotations of the factory, and the errors."""
        return _resolve_parameter_hints(self.factory)

    def provided_type(self, name: str) -> object:
        """Resolve the return annotation of the factory of service *name*.

        The build compares it with the handler parameter's annotation. A
        return annotation that does not resolve raises ``TypeError``.
        """
        source = _annotation_source(self.factory)
        hints, errors = _resolve_parameter_hints(self.factory)
        if "return" in errors:
            raise TypeError(
                f"RunScoped factory {_factory_label(source)!r} for service "
                f"{name!r} has a "
                f"return annotation that did not resolve ({errors['return']}). "
                f"The framework compares it with the annotation of each handler "
                f"or factory parameter that reads it, at graph build, so it "
                f"cannot be left unresolvable. "
                f"Make every name in the annotation importable at run time, or "
                f"declare the class at module level."
            )
        return _resolve_self(hints["return"], source)


class _FactoryBinder:
    """Bind each ``RunScoped`` factory of a name-keyed ``services=`` to its
    inputs, for one graph. A factory parameter binds by the rules of a
    handler parameter."""

    def __init__(
        self,
        services: Services,
        readables: Mapping[str, Readable],
        context_schema: type | None,
    ) -> None:
        self._services = services
        self._readables = readables
        self._context_schema = context_schema
        self._bound: dict[str, BoundFactory] = {}

    def bind_all(self) -> dict[str, BoundFactory]:
        for name, value in self._services.by_name.items():
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
        scoped = cast("RunScoped[Any]", self._services.by_name[name])
        label = scoped._label(name)
        hints, errors = scoped._hints()
        params: list[Param] = []
        missing: list[ServiceCandidate] = []
        positional = 0
        for param in scoped._parameters(name):
            p = param.name
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"{label} has the variadic parameter {p!r}. Each parameter "
                    f"must name one input. Replace it with one parameter per "
                    f"input."
                )
            positional += param.kind is param.POSITIONAL_ONLY
            hint = resolved_hint(label, p, hints, errors)
            source = classify(
                label, p, hint, self._readables.get, self._context_schema
            ) or self._services._service_input(
                label, p, hint, lambda dep: self._bind(dep, (*path, name))
            )
            if source is None:
                missing.append(ServiceCandidate(p, hint, param.default))
            else:
                params.append(Param(p, source, param.default))
        if missing:
            refuse_unclaimed(
                label,
                missing,
                reducer_names=self._readables.keys(),
                received=self._services.received(without=name),
                entry="the graph",
                context_schema=self._context_schema,
                reader="factory",
            )
        bound = BoundFactory(name, scoped.factory, label, tuple(params), positional)
        self._bound[name] = bound
        return bound


class Services:
    """The services of a graph: each typed field of the ``services=`` object.

    The class of the object declares one field per service. The field type
    is the declared type of the service. A ``RunScoped[T]`` field holds a
    factory whose result has the type ``T``. A reader names the declared
    type exactly.
    """

    def __init__(
        self,
        values: Mapping[str, object],
        declared: Mapping[str, object],
        factories: Mapping[str, BoundFactory] | None = None,
    ) -> None:
        """*factories* holds the bound ``RunScoped`` factories, once
        :meth:`bind_factories` has bound them for one graph."""
        self.by_name = dict(values)
        self._declared = dict(declared)
        self._factories = dict(factories or {})

    @classmethod
    def parse(cls, services: object) -> Services:
        """Read the services of a ``services=`` object. ``None`` holds none.

        Raises ``TypeError`` for an object whose class declares no typed
        field, such as a dict or a list, with the fix.
        """
        if services is None:
            return cls({}, {})
        fields = _declared_fields(services)
        declared: dict[str, object] = {}
        for name, hint in fields.items():
            value = getattr(services, name)
            declared[name] = _declared_type(name, hint, value)
        return cls({n: getattr(services, n) for n in fields}, declared)

    @property
    def factories(self) -> Mapping[str, BoundFactory]:
        """Each bound ``RunScoped`` factory, by service name."""
        return self._factories

    def received(self, without: str | None = None) -> str:
        """State the services that ``services=`` holds, for an error.
        *without* is a service to leave out: the factory that the error
        names."""
        names = sorted(n for n in self.by_name if n != without)
        if not names:
            return (
                "services= has no other service."
                if without
                else "services= has no service."
            )
        return f"services= has the fields {names}."

    def bind_factories(
        self, readables: Mapping[str, Readable], context_schema: type | None
    ) -> Services:
        """These services, with each ``RunScoped`` factory bound to its
        inputs by the handler rules.

        Raises ``TypeError`` for a service whose name a reducer or a
        ``Derived`` also has, because a parameter cannot name both. Raises
        ``TypeError`` for each factory parameter that cannot bind, and for a
        cycle among factories.
        """
        for name in self.by_name:
            readable = readables.get(name)
            if readable is not None:
                kind = "reducer" if isinstance(readable, BaseReducer) else "Derived"
                raise TypeError(
                    f"services= has the field {name!r}, and a {kind} has the "
                    f"same name. A parameter cannot name both. Rename the "
                    f"service or the {kind}."
                )
        binder = _FactoryBinder(self, readables, context_schema)
        return Services(self.by_name, self._declared, binder.bind_all())

    def bind(self, meta: HandlerMeta, context_schema: type | None) -> BoundHandler:
        """Fill each candidate of *meta* with the service of its name, and
        check the annotation against the declared type of the service. A
        ``RunScoped`` service fills its candidate with its bound factory."""
        services: list[Param] = []
        for c in meta.service_candidates:
            source = self._service_input(
                meta.label, c.param, c.annotation, self._factories.__getitem__
            )
            if source is not None:
                services.append(Param(c.param, source, c.default))
        return BoundHandler(meta, tuple(services))

    def _service_input(
        self,
        owner: str,
        param: str,
        hint: object,
        bind_factory: Callable[[str], BoundFactory],
    ) -> ValueInput | BoundFactory | None:
        """The service of name *param*, or ``None``. The annotation *hint*
        must name the declared type of the service exactly. *bind_factory*
        gives the bound factory of a ``RunScoped`` name."""
        if param not in self.by_name:
            return None
        require_same_type(
            owner, param, hint, self._declared[param], f"service {param!r}", False
        )
        value = self.by_name[param]
        if isinstance(value, RunScoped):
            return bind_factory(param)
        return ValueInput(value)


def _declared_fields(services: object) -> dict[str, object]:
    """The typed fields that the class of *services* declares, by name.

    Raises ``TypeError`` for an object with no typed field, and for an
    annotation that does not resolve.
    """
    kind = type(services)
    fix = (
        "services= takes an object whose class declares one typed field per "
        "service, such as a dataclass instance. Got a "
        f"{kind.__name__}. Declare a dataclass with one field per service, "
        "typed with the service type, or RunScoped[T] for a factory, and pass "
        "an instance of it."
    )
    if isinstance(services, tuple) and hasattr(kind, "_fields"):
        raise TypeError(
            f"services= is a NamedTuple {kind.__name__}, which is a tuple of "
            f"values by position. Declare the services in a class with typed "
            f"fields instead: use a dataclass with the same fields."
        )
    if isinstance(services, (Mapping, list, tuple, set, frozenset, str, bytes)):
        raise TypeError(fix)
    try:
        hints = typing.get_type_hints(kind)
    except Exception as exc:
        raise TypeError(
            f"services= is a {kind.__name__} whose field annotations do not "
            f"resolve ({type(exc).__name__}: {exc}). Make every name in them "
            f"importable at run time."
        ) from exc
    fields: dict[str, object] = {}
    for name, hint in hints.items():
        if typing.get_origin(hint) is typing.ClassVar or hint is typing.ClassVar:
            continue
        if isinstance(hint, dataclasses.InitVar) or hint is dataclasses.InitVar:
            raise TypeError(
                f"services= declares the InitVar field {name!r}, which the "
                f"object does not keep, so it cannot be a service. Make it a "
                f"field, or move it out of the services object."
            )
        final = typing.get_origin(hint) is typing.Final
        fields[name] = typing.get_args(hint)[0] if final else hint
    if not fields:
        raise TypeError(fix)
    for name in fields:
        if not hasattr(services, name):
            raise TypeError(
                f"services= declares the field {name!r}, but the object has no "
                f"value for it. Give it a value."
            )
    return fields


def _declared_type(name: str, hint: object, value: object) -> object:
    """The type that readers of service *name* must name: the field type,
    or ``T`` for a ``RunScoped[T]`` field.

    Raises ``TypeError`` when a ``RunScoped`` value and its field type do
    not agree, or when the factory returns another type than ``T``.
    """
    try:
        _normal(hint)
    except UnsettledTypeError as exc:
        raise TypeError(
            f"Service {name!r} is declared {_hint_label(hint)}, but {exc}. "
            f"Declare it with a class."
        ) from None
    scoped = typing.get_origin(hint) is RunScoped or hint is RunScoped
    if isinstance(value, RunScoped) != scoped:
        raise TypeError(
            f"Service {name!r} is declared {_hint_label(hint)}, but its value "
            f"is {'a' if isinstance(value, RunScoped) else 'not a'} RunScoped "
            f"factory. Type a factory field RunScoped[T], and any other field "
            f"with the type of its value."
        )
    if not scoped:
        return hint
    arguments = typing.get_args(hint)
    if not arguments:
        raise TypeError(
            f"Service {name!r} is declared RunScoped with no type. Declare the "
            f"type of its result: RunScoped[T]."
        )
    declared = arguments[0]
    returns = cast("RunScoped[Any]", value).provided_type(name)
    try:
        same = _same_type(returns, declared, may_be_unset=False)
    except UnsettledTypeError as exc:
        raise TypeError(
            f"Service {name!r} is declared RunScoped[{_hint_label(declared)}], "
            f"but the return annotation of its factory {exc}. Make the factory "
            f"return {_hint_label(declared)}."
        ) from None
    if not same:
        raise TypeError(
            f"Service {name!r} is declared RunScoped[{_hint_label(declared)}], "
            f"but its factory returns {_hint_label(returns)}. Make the factory "
            f"return {_hint_label(declared)}, or change the field type."
        )
    return declared


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


def _factory_label(factory: object) -> str:
    """Render a ``RunScoped`` factory for an error message."""
    if inspect.isroutine(factory):
        return factory.__qualname__
    return _hint_label(factory)
