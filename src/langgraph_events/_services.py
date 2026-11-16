"""Service markers for ``EventGraph(services=...)``."""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from langgraph_events._handler import (
    BoundHandler,
    ServiceCandidate,
    refuse_unclaimed,
)
from langgraph_events._hints import (
    _annotation_source,
    _hint_fits,
    _hint_label,
    _open_parameters,
    _resolve_hints_and_errors,
    _resolve_parameter_hints,
    _satisfies,
)
from langgraph_events._inputs import (
    BoundFactory,
    Param,
    ValueInput,
    check_fit,
    classify,
    resolved_hint,
)
from langgraph_events._reducer import BaseReducer

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
class RunScoped(Generic[T]):
    """A service that a factory builds once per node call.

    Place a ``RunScoped`` value in the name-keyed ``services=`` mapping. Its
    result is injected under the handler's parameter name.

    Each parameter of the factory binds by the rules of a handler
    parameter. An annotation of exactly ``RunnableConfig``, the graph's
    ``context_schema``, ``EventLog``, ``BaseStore`` or ``Reflection`` gets
    that input. Each other parameter gets the stored reducer, the
    ``Derived`` or the name-keyed service of its name, and its annotation
    must fit that value. Each parameter must have an annotation that
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

        language = ScalarReducer(
            name="language", event_type=LanguageSet, fn=lambda e: e.language
        )
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
        if isinstance(source, type):
            return source
        hints, errors = _resolve_hints_and_errors(source)
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
        services: NamedServices,
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
        scoped: RunScoped[Any] = self._services.by_name[name]
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
                entry="services=",
                context_schema=self._context_schema,
                reader="factory",
            )
        bound = BoundFactory(name, scoped.factory, label, tuple(params), positional)
        self._bound[name] = bound
        return bound


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

    @property
    def factories(self) -> Mapping[str, BoundFactory]:
        """The type-keyed form holds no ``RunScoped`` factory."""
        return {}

    def bind_factories(
        self, readables: Mapping[str, Readable], context_schema: type | None
    ) -> TypedServices:
        """The type-keyed form holds no ``RunScoped`` factory."""
        return self

    def bind(self, meta: HandlerMeta, context_schema: type | None) -> BoundHandler:
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
        detected: list[Param] = []
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
                detected.append(_value_param(candidate, self._by_type[hint]))
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
                detected.append(_value_param(candidate, self._by_type[matches[0]]))
        return BoundHandler(meta, tuple(detected))


class NamedServices:
    """The mapping form of ``services=``: each service fills the handler
    parameter of its name, so two instances of one type can coexist (a
    primary and a backup chat model). The value must fit the annotation.
    """

    def __init__(
        self,
        services: Mapping[str, Any],
        factories: Mapping[str, BoundFactory] | None = None,
    ) -> None:
        """*factories* holds the bound ``RunScoped`` factories, once
        :meth:`bind_factories` has bound them for one graph."""
        self.by_name = dict(services)
        # The type that each ``RunScoped`` factory returns. Resolved here, so
        # a bad return annotation does not wait for the first handler.
        self._run_scoped_types = {
            name: value.provided_type(name)
            for name, value in self.by_name.items()
            if isinstance(value, RunScoped)
        }
        self._factories = dict(factories or {})

    @property
    def factories(self) -> Mapping[str, BoundFactory]:
        """Each bound ``RunScoped`` factory, by service name."""
        return self._factories

    def received(self, without: str | None = None) -> str:
        """State the name keys that ``services=`` holds, for an error.
        *without* is a key to leave out: the factory that the error names."""
        names = sorted(n for n in self.by_name if n != without)
        if not names:
            return (
                "services= has no other service."
                if without
                else ("services= has no service.")
            )
        return f"services= has the name keys {names}."

    def bind_factories(
        self, readables: Mapping[str, Readable], context_schema: type | None
    ) -> NamedServices:
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
                    f"services= has the name key {name!r}, and a {kind} has the "
                    f"same name. A parameter cannot name both. Rename the "
                    f"service or the {kind}."
                )
        binder = _FactoryBinder(self, readables, context_schema)
        return NamedServices(self.by_name, binder.bind_all())

    def bind(self, meta: HandlerMeta, context_schema: type | None) -> BoundHandler:
        """Fill each candidate of *meta* with the service of its name, and
        check the service against the annotation. A ``RunScoped`` service
        fills its candidate with its bound factory."""
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
        """The service of name *param*, checked against *hint*, or ``None``.

        A plain value must be an instance of *hint*. A ``RunScoped`` factory
        must return a subtype of *hint*. *bind_factory* gives the bound
        factory of a ``RunScoped`` name. The comparison uses
        ``beartype.door``. It handles ``Union``, a runtime ``Protocol`` with
        methods only, and a generic alias. A ``NewType`` is nominal for a
        factory and erased for a plain value.
        """
        if param not in self.by_name:
            return None
        value = self.by_name[param]
        if isinstance(value, RunScoped):
            provided = self._run_scoped_types[param]
            factory = _factory_label(_annotation_source(value.factory))
            returns = _hint_label(provided)
            check_fit(
                owner,
                param,
                hint,
                _hint_fits(provided, hint),
                provided,
                f"services[{param!r}] is RunScoped({factory}) and that factory "
                f"returns {returns}",
                f"Change the annotation to {returns}, or register a factory "
                f"that returns {_hint_label(hint)}.",
            )
            return bind_factory(param)
        # A cold import costs about 200 ms. Only the first graph in the
        # process with a name-keyed service pays it.
        from beartype.door import is_bearable  # noqa: PLC0415

        label = _hint_label(hint)
        if value is None:
            what = f"services[{param!r}] is None"
            widen = f"Widen the annotation to {label} | None"
        else:
            what = f"services[{param!r}] is an instance of {type(value).__qualname__}"
            widen = "Widen the annotation"
        check_fit(
            owner,
            param,
            hint,
            _satisfies(is_bearable, value, hint),
            type(value),
            f"{what}, not an instance of {label}",
            f"{widen}, or register a value that satisfies it.",
        )
        return ValueInput(value)


def _value_param(candidate: ServiceCandidate, value: object) -> Param:
    """The parameter of *candidate*, filled with the plain service *value*."""
    return Param(candidate.param, ValueInput(value), candidate.default)


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


def _factory_label(factory: object) -> str:
    """Render a ``RunScoped`` factory for an error message."""
    if inspect.isroutine(factory):
        return factory.__qualname__
    return _hint_label(factory)
