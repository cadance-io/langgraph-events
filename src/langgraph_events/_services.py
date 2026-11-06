"""Service markers for ``EventGraph(services=...)``."""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from langgraph_events._handler import BoundHandler, _resolve_hints_and_errors
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.runnables import RunnableConfig

    from langgraph_events._handler import HandlerMeta

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
    """A service derived per run from the node's ``RunnableConfig``.

    Place a ``RunScoped`` value in the name-keyed ``services=`` mapping. The
    factory is called with the run config once per node call. Its result is
    injected under the handler's parameter name.

    A factory whose second parameter is annotated with the graph's
    ``context_schema`` also receives the run context, as
    ``factory(config, context)``. That parameter must be positional and
    required. The graph build refuses any other second parameter.

    The factory must be a plain callable. A coroutine function is rejected,
    because its result would be injected without an ``await``.

    Example::

        services = {"model": RunScoped(model_for)}

        @on(SomeEvent)
        def handle(event: SomeEvent, model: ConversationModel) -> None: ...
    """

    factory: Callable[[RunnableConfig], T] | Callable[[RunnableConfig, Any], T]

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

    def takes_context(self, name: str, context_schema: type | None) -> bool:
        """Whether the factory of service *name* takes the run context.

        It does when its second parameter is positional, required and
        annotated with *context_schema*. A signature that Python cannot read
        counts as the run config only. Raises ``TypeError`` for any other
        second parameter, with the fix.
        """
        try:
            params = inspect.signature(self.factory).parameters.values()
        except (TypeError, ValueError):
            return False
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        extra = [p for p in params if p.kind not in variadic][1:]
        if not extra:
            return False
        source = _annotation_source(self.factory)
        label = f"RunScoped factory {_factory_label(source)!r} for service {name!r}"
        second = extra[0]
        positional = second.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        required = second.default is inspect.Parameter.empty
        if context_schema is None:
            if not (positional and required):
                return False
            raise TypeError(
                f"{label} takes a second parameter {second.name!r}, but the graph "
                f"has no context_schema. Pass context_schema= to the graph to give "
                f"the factory the run context, or give the factory one parameter."
            )
        init = source.__init__ if isinstance(source, type) else source  # type: ignore[misc]
        hints, _ = _resolve_hints_and_errors(init)
        annotation = hints.get(second.name, inspect.Parameter.empty)
        if len(extra) == 1 and positional and required and annotation is context_schema:
            return True
        if len(extra) > 1:
            why = "has more than two parameters"
        elif not positional:
            why = f"has a keyword-only parameter {second.name!r}"
        elif not required:
            why = f"has an optional parameter {second.name!r}"
        elif annotation is inspect.Parameter.empty:
            why = f"has an unannotated parameter {second.name!r}"
        else:
            why = f"annotates parameter {second.name!r} as {_hint_label(annotation)}"
        schema = context_schema.__qualname__
        raise TypeError(
            f"{label} {why}. A factory that takes the run context declares it "
            f"as a second positional parameter, required and annotated with "
            f"{schema}: (config, {second.name}: {schema}). Change the parameter, "
            f"or remove it."
        )

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
        if context_schema is not None:
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

    def bind(self, meta: HandlerMeta, context_schema: type | None) -> BoundHandler:
        """Fill each candidate of *meta* with the service of its name, then
        check each service against the annotation. A ``RunScoped`` factory
        whose second parameter is annotated with *context_schema* also gets
        the run context."""
        services = tuple(
            (c.param, self._by_name[c.param])
            for c in meta.service_candidates
            if c.param in self._by_name
        )
        bound = BoundHandler(
            meta,
            services,
            frozenset(
                param
                for param, value in services
                if isinstance(value, RunScoped)
                and value.takes_context(param, context_schema)
            ),
        )
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
