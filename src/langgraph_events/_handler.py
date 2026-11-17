"""The ``@on`` decorator and handler metadata extraction."""

import annotationlib
import inspect
import typing
from collections import abc
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

from langgraph_events._event import Event, Invariant
from langgraph_events._hints import _resolve_hints_and_errors
from langgraph_events._identity import command_identity
from langgraph_events._inputs import Param, classify, resolved_hint
from langgraph_events._retry import RetryPolicy
from langgraph_events._validate import normalize_exception_tuple
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from langgraph_events._reducer import Readable
    from langgraph_events._types import F, HandlerReturn


def _validate_invariants(
    invariants: dict[type[Invariant], Callable[..., bool]] | None,
) -> tuple[tuple[type[Invariant], Callable[..., bool]], ...]:
    """Validate and normalise the @on(invariants=) argument.

    Accepts a dict mapping an ``Invariant`` subclass (used as the dispatch key
    via ``InvariantViolated.invariant``) to a sync predicate that takes an
    ``EventLog`` and returns bool. Predicate exceptions propagate at dispatch
    — they are not turned into violations.
    """
    if not invariants:
        return ()
    if not isinstance(invariants, dict):
        raise TypeError(
            f"@on() invariants= must be a dict[type[Invariant], Callable], "
            f"got {type(invariants).__name__}"
        )
    validated: list[tuple[type[Invariant], Callable[..., bool]]] = []
    for inv_cls, pred in invariants.items():
        if not (isinstance(inv_cls, type) and issubclass(inv_cls, Invariant)):
            raise TypeError(
                f"@on() invariants= keys must be Invariant subclasses, got {inv_cls!r}"
            )
        if not callable(pred):
            raise TypeError(
                f"@on() invariants= predicate for {inv_cls.__name__!r} must "
                f"be callable, got {pred!r}"
            )
        if inspect.iscoroutinefunction(pred):
            raise TypeError(
                f"@on() invariants= predicate for {inv_cls.__name__!r} must "
                f"be sync, got async function {pred.__qualname__!r}"
            )
        try:
            inv_cls()
        except TypeError as exc:
            raise TypeError(
                f"@on() invariants= Invariant subclass {inv_cls.__name__!r} "
                f"must be zero-arg instantiable; the framework calls "
                f"{inv_cls.__name__}() at violation time purely for "
                f"isinstance matching. Remove required fields from the "
                f"subclass body. Got: {exc}"
            ) from exc
        validated.append((inv_cls, pred))
    return tuple(validated)


def _resolve_type_hints(fn: Callable[..., object]) -> dict[str, Any]:
    """Return ``fn``'s resolvable type hints. Unresolvable ones are absent."""
    return _resolve_hints_and_errors(fn)[0]


def _infer_event_type(fn: Callable[..., object]) -> type[Event]:
    """Read an ``Event`` subclass off ``fn``'s first parameter annotation.

    Used by ``@on`` when positional event types are omitted. Raises
    ``TypeError`` with an actionable message for the full range of failure
    modes (missing parameter, missing annotation, non-``Event`` type, Union).
    """
    hints, hint_errors = _resolve_hints_and_errors(fn)
    sig = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    params = [p for p in sig.parameters if p != "self"]
    if not params:
        raise TypeError(
            f"@on requires {fn.__qualname__!r} to declare a typed first "
            f"parameter (the event), but it has none."
        )
    first = params[0]
    if first in hint_errors:
        raise TypeError(
            f"@on could not resolve the annotation on {fn.__qualname__!r}'s "
            f"first parameter {first!r} ({hint_errors[first]}). Make the "
            f"annotation importable at run time, or pass the event type "
            f"explicitly: @on(EventType)."
        )
    event_type = hints.get(first)
    if event_type is None:
        raise TypeError(
            f"@on requires {fn.__qualname__!r}'s first parameter {first!r} "
            f"to be annotated with an Event subclass (got no annotation), "
            f"or pass the event type explicitly: @on(EventType)."
        )
    if not isinstance(event_type, type):
        # Catches X | Y unions and other non-class annotations.
        raise TypeError(
            f"@on requires {fn.__qualname__!r}'s first parameter {first!r} "
            f"to be annotated with a single Event subclass, got "
            f"{event_type!r}. For multi-event subscription pass the types "
            f"explicitly: @on(A, B, ...)."
        )
    is_event_type = issubclass(event_type, Event) or getattr(
        event_type, "_event_mixin", False
    )
    if not is_event_type:
        raise TypeError(
            f"@on requires {fn.__qualname__!r}'s first parameter {first!r} "
            f"to be annotated with an Event subclass or mixin, got "
            f"{event_type.__name__}."
        )
    return event_type


def normalize_previous_names(previously: object, *, owner: str) -> tuple[str, ...]:
    """Validate and normalise a ``previously=`` value into an alias tuple.

    *owner* prefixes error messages with the declaration site's voice —
    ``"@on()"`` for the decorator, ``"Command 'X'"`` for the class attribute —
    so the user is pointed at the spelling they actually wrote. Only ``str``
    and re-readable sequences are accepted: a generator would be exhausted on
    the first graph build and silently yield no aliases on the next.
    """
    if isinstance(previously, str):
        previous_names: tuple[str, ...] = (previously,)
    elif isinstance(previously, abc.Sequence):
        previous_names = tuple(previously)
    else:
        raise TypeError(
            f"{owner} previously= must be a str or a sequence of str, "
            f"got {previously!r}"
        )
    for alias in previous_names:
        if not isinstance(alias, str) or not alias.strip():
            raise TypeError(
                f"{owner} previously= node names must be non-empty str, got {alias!r}"
            )
    return previous_names


def _build_on_decorator(
    event_types: tuple[type[Event], ...],
    *,
    raises: type[Exception] | tuple[type[Exception], ...] = (),
    invariants: dict[type[Invariant], Callable[..., bool]] | None = None,
    field_matchers: dict[str, type[Event] | type[Exception] | type[Invariant] | str]
    | None = None,
    node_name: str | None = None,
    previous_names: tuple[str, ...] = (),
    retry: RetryPolicy | None = None,
) -> Callable[[F], F]:
    """Validate arguments and return the decorator that stamps attributes.

    Modifiers are keyword-only: they are forwarded from three call sites in
    ``on()``, and every new modifier would otherwise have to be threaded
    positionally through all of them in lockstep.
    """
    field_matchers = field_matchers or {}
    for et in event_types:
        if not (
            isinstance(et, type)
            and (issubclass(et, Event) or getattr(et, "_event_mixin", False))
        ):
            raise TypeError(f"@on() requires Event subclasses or mixins, got {et!r}")

    raises_tuple = normalize_exception_tuple(raises, owner="@on() raises=")

    if retry is not None and not isinstance(retry, RetryPolicy):
        raise TypeError(
            f"@on() retry= must be a RetryPolicy instance, got {retry!r}. "
            f"Import it as 'from langgraph_events import RetryPolicy' — note "
            f"this is not langgraph.types.RetryPolicy, which is node-level."
        )

    invariants_tuple = _validate_invariants(invariants)

    # Validate field matchers: Event/Exception subclass for type-based isinstance
    # match, or a bare str for equality match on string fields. Non-Exception
    # BaseException subclasses (KeyboardInterrupt, SystemExit, GeneratorExit,
    # asyncio.CancelledError) are rejected for symmetry with raises= — the
    # framework treats them as runtime/exit signals, not domain errors.
    for field_name, field_match in field_matchers.items():
        is_type = isinstance(field_match, type) and (
            issubclass(field_match, Event)
            or issubclass(field_match, Exception)
            or issubclass(field_match, Invariant)
        )
        is_str = isinstance(field_match, str)
        if not (is_type or is_str):
            raise TypeError(
                f"@on() field matcher values must be an Event, Exception, or "
                f"Invariant subclass, or a str (for equality match), got "
                f"{field_match!r} for field {field_name!r}"
            )
        # Check that at least one event type declares this field
        has_field = any(
            field_name in getattr(et, "__dataclass_fields__", {}) for et in event_types
        )
        if not has_field:
            type_names = ", ".join(t.__name__ for t in event_types)
            raise TypeError(
                f"@on() field matcher references {field_name!r}, but "
                f"no field {field_name!r} exists on ({type_names})"
            )
        # `invariant=` matches against InvariantViolated.invariant (always
        # an Invariant instance). A string value would never fire at
        # runtime — reject it at decoration time so the no-op is caught
        # before it silently ships.
        if field_name == "invariant" and not is_type:
            raise TypeError(
                f"@on() invariant= must reference an Invariant subclass, "
                f"not {field_match!r}; InvariantViolated.invariant is "
                f"always an Invariant instance and a string matcher would "
                f"never fire."
            )

    def decorator(fn: F) -> F:
        fn._event_types = event_types  # type: ignore[attr-defined]
        if field_matchers:
            fn._field_matchers = dict(field_matchers)  # type: ignore[attr-defined]
        # Unconditional, all four: every EventGraph build re-stamps inline
        # command handlers from the class's *current* ``raises`` /
        # ``invariants`` / ``previously`` / ``retry``, so a conditional stamp
        # would let an earlier build's value keep a since-removed contract,
        # alias, or policy alive. Their falsy values match the defaults
        # ``HandlerMeta.of`` reads them with, so an empty stamp is
        # indistinguishable from no stamp for every consumer.
        fn._raises = raises_tuple  # type: ignore[attr-defined]
        fn._invariants = invariants_tuple  # type: ignore[attr-defined]
        fn._previous_names = previous_names  # type: ignore[attr-defined]
        fn._retry = retry  # type: ignore[attr-defined]
        # ``node_name`` has no class-level declaration to re-stamp from (see
        # ``HandlerMeta.of``), so there is no stale value to correct and
        # an unconditional stamp would only erase an explicit
        # ``@on(node_name=...)`` pin — the identity that keeps interrupted
        # checkpoints resumable across a rename.
        if node_name is not None:
            fn._node_name = node_name  # type: ignore[attr-defined]
        return fn

    return decorator


def on(
    *event_types: Any,
    raises: type[Exception] | tuple[type[Exception], ...] = (),
    invariants: dict[type[Invariant], Callable[..., bool]] | None = None,
    node_name: str | None = None,
    previously: str | tuple[str, ...] = (),
    retry: RetryPolicy | None = None,
    **field_matchers: type[Event] | type[Exception] | type[Invariant] | str,
) -> Any:  # noqa: ANN401 - bare @on returns the function, @on(...) returns a decorator
    """Subscribe a handler to one or more event types.

    Three shapes, escalating by what's needed:

    1. **Bare** — ``@on`` (no parens). Infers the event type from the
       handler's first parameter annotation::

           @on
           def place(event: Order.Place) -> Order.Place.Placed:
               return Order.Place.Placed(order_id="o1")

    2. **Modifiers only** — ``@on(raises=..., retry=..., invariants=..., field=...)``.
       Infers the event type from the annotation and applies modifiers::

           @on(invariants={CustomerNotBanned: lambda log: ...})
           def place(event: Order.Place) -> Order.Place.Placed: ...

    3. **Explicit types** — ``@on(EventA, EventB, ...)``. Required for
       multi-event subscription or when you prefer not to rely on the
       annotation::

           @on(UserMessage, ToolResults)
           async def call_llm(event: Event) -> AssistantMessage: ...

    ``raises=`` declares exception classes the framework should catch from
    this handler; a matching ``@on(HandlerRaised, exception=...)`` catcher
    must exist at compile time.

    ``retry=RetryPolicy(...)`` wraps the handler call in declarative backoff:
    the framework re-invokes it in place on a declared raise, and
    ``HandlerRaised`` fires only once the budget is spent — or, earlier, once
    the next backoff would cross the run's ``deadline=``, in which case it is
    a ``HandlerAbandonedOnDeadline``. When the run's ``pause_on=`` signal is
    set, the give-up is a ``HandlerAbandonedOnSignal``. Requires a non-empty
    ``raises=``; the policy's ``on=`` must overlap it.

    Field matchers narrow dispatch — ``@on(Resumed, interrupted=Approval)``
    for ``isinstance`` match (works for Event, Exception, or Invariant
    subclasses); string values do equality match (e.g. a string event field).

    ``node_name=`` pins the handler's graph-node identity (default: the
    function name) so renaming the function never breaks an interrupted
    checkpoint; ``previously=`` (str or tuple) declares historic node names to
    keep resumable after a rename. These and ``retry=`` are reserved keywords —
    a field named ``node_name``, ``previously``, or ``retry`` cannot be matched
    positionally via ``**field_matchers``. Inline ``Command`` handlers declare
    historic node names as a class attribute instead:
    ``previously: ClassVar = (...)``.
    """
    if node_name is not None and not isinstance(node_name, str):
        raise TypeError(f"@on() node_name= must be a str, got {node_name!r}")
    previous_names = normalize_previous_names(previously, owner="@on()")

    no_modifiers = (
        raises == ()
        and invariants is None
        and not field_matchers
        and node_name is None
        and not previous_names
        and retry is None
    )
    sole_arg_is_function = len(event_types) == 1 and (
        inspect.isfunction(event_types[0]) or inspect.ismethod(event_types[0])
    )

    if sole_arg_is_function and no_modifiers:
        fn = event_types[0]
        return _build_on_decorator((_infer_event_type(fn),))(fn)

    if not event_types:

        def inferring(fn: F) -> F:
            return _build_on_decorator(
                (_infer_event_type(fn),),
                raises=raises,
                invariants=invariants,
                field_matchers=dict(field_matchers),
                node_name=node_name,
                previous_names=previous_names,
                retry=retry,
            )(fn)

        return inferring

    return _build_on_decorator(
        event_types,
        raises=raises,
        invariants=invariants,
        field_matchers=dict(field_matchers),
        node_name=node_name,
        previous_names=previous_names,
        retry=retry,
    )


class ServiceRequirement(NamedTuple):
    """A handler parameter that only a service can fill.

    The event, a framework injectable, a reducer and a field matcher do not
    claim it. :meth:`EventGraph.of` must bind a service to it, or the build
    raises ``TypeError``.
    """

    node: str
    """The graph-node name of the handler: its checkpoint identity."""
    qualname: str
    """The module and qualified name of the handler function."""
    param: str
    """The parameter name."""
    annotation: Any
    """The resolved annotation. A handler parameter must have one."""


class ServiceCandidate(NamedTuple):
    """A parameter that the event, the injectables, the reducers and the
    field matchers do not claim."""

    param: str
    annotation: Any
    """The resolved annotation."""
    default: object = inspect.Parameter.empty
    """The default of the parameter, or ``inspect.Parameter.empty``."""


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler, with no services.

    The ``services=`` of one graph bind to its service candidates.
    """

    name: str
    fn: Callable[..., HandlerReturn]
    event_types: tuple[type[Event], ...]
    is_async: bool
    # Stable graph-node / checkpoint identity. Equals ``name`` for ordinary
    # handlers, but for inline ``Command.handle()`` handlers it is the command's
    # ``__qualname__`` (e.g. ``Order.Place``) so the node a paused checkpoint
    # resumes into never depends on ``handlers=[...]`` order. ``name`` stays the
    # human-readable handler label used in choreography/mermaid/diagnostics.
    # Required — always populated by ``HandlerMeta.of``; never empty.
    node_name: str
    # Historic node names this handler answered to, declared via
    # ``@on(previously=...)``. The graph registers an alias node per name so an
    # interrupted checkpoint paused at the old node still resumes after a rename.
    previous_names: tuple[str, ...] = ()
    # Each parameter that a framework input, a stored reducer or a
    # ``Derived`` fills, in signature order. A service fills the others.
    params: tuple[Param, ...] = ()
    # Each entry is (field_name, matcher, is_type_matcher). The bool is
    # precomputed at extract time so the hot-path ``matches`` loop avoids
    # an ``isinstance(matcher, type)`` probe per dispatch.
    field_matchers: tuple[
        tuple[str, type[Event] | type[Exception] | type[Invariant] | str, bool],
        ...,
    ] = ()
    field_inject_params: frozenset[str] = frozenset()
    raises: tuple[type[Exception], ...] = ()
    # Declared via ``@on(retry=...)`` or a ``retry`` class attribute on a
    # Command. ``None`` means one attempt, the pre-retry behaviour.
    retry: RetryPolicy | None = None
    invariants: tuple[tuple[type[Invariant], Callable[..., bool]], ...] = ()
    # Each non-variadic param that the event, ``params`` or a field matcher
    # does not claim, in signature order. Only a service can fill one. The
    # ``services=`` of one graph bind to these.
    service_candidates: tuple[ServiceCandidate, ...] = ()

    @property
    def qualname(self) -> str:
        """The module and qualified name of the handler function."""
        return f"{self.fn.__module__}.{self.fn.__qualname__}"

    @property
    def label(self) -> str:
        """The subject of a construction-time error or warning about this
        handler."""
        return f"Handler '{self.qualname}' (node '{self.node_name}')"

    @property
    def requirements(self) -> tuple[ServiceRequirement, ...]:
        """Each parameter that only a service can fill, in signature order."""
        return tuple(
            ServiceRequirement(self.node_name, self.qualname, c.param, c.annotation)
            for c in self.service_candidates
        )

    @classmethod
    def of(
        cls,
        fn: Callable[..., Any],
        *,
        readables: Mapping[str, Readable],
        context_schema: type | None = None,
    ) -> HandlerMeta:
        """Extract the metadata of a decorated function, with no services.

        Each parameter after the event binds by the rules of
        :func:`classify`, like a ``RunScoped`` factory parameter. Raises
        ``TypeError`` for an annotation that does not name the declared type
        of its reducer or ``Derived``. Each other non-variadic parameter that
        no field matcher claims is recorded in ``service_candidates``. The
        ``services=`` of one graph bind to them.
        """
        event_types = getattr(fn, "_event_types", None)
        if event_types is None:
            raise ValueError(
                f"Function {fn.__qualname__!r} is not decorated with @on(EventType)"
            )

        hints, hint_errors = _resolve_hints_and_errors(fn)

        name = getattr(fn, "_node_name", None) or fn.__name__
        # Inline command handlers route by their command's qualname (stable,
        # order-independent); every other handler routes by its own name. The
        # qualname intentionally wins over any ``_node_name`` here — inline
        # handlers never carry one today (``_expand_command_handlers`` forwards
        # raises/invariants/previously but no ``node_name``), but were that to
        # change, command identity must stay the resume identity.
        inline_command = getattr(fn, "_inline_command", None)
        node_name = (
            command_identity(inline_command) if inline_command is not None else name
        )
        label = f"Handler '{fn.__module__}.{fn.__qualname__}' (node '{node_name}')"
        # Each annotation must resolve, also on the event and a field matcher.
        for pname in hint_errors.keys() - {"return"}:
            resolved_hint(label, pname, hints, hint_errors)

        # Extract field matchers; classify each now so dispatch avoids isinstance.
        raw_field_matchers: dict[
            str, type[Event] | type[Exception] | type[Invariant] | str
        ] = getattr(fn, "_field_matchers", {})
        field_matchers = tuple(
            (name, matcher, isinstance(matcher, type))
            for name, matcher in raw_field_matchers.items()
        )
        sig = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
        field_inject_params = frozenset(
            name for name in sig.parameters if name in raw_field_matchers
        )

        # The first (event) param and each field matcher are claimed. Variadic
        # params (``*args`` / ``**kwargs``) cannot be claimed by any injection
        # source; they are caller-controlled. They are no candidate, so
        # generic catcher-style handlers (``def react(event, *a, **kw)``) build.
        first_param = next(iter(sig.parameters), None)
        claimed = {first_param, "self", *raw_field_matchers}
        variadic = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        params: list[Param] = []
        service_candidates: list[ServiceCandidate] = []
        for pname, param in sig.parameters.items():
            if pname in claimed or param.kind in variadic:
                continue
            hint = resolved_hint(label, pname, hints, hint_errors)
            source = classify(label, pname, hint, readables.get, context_schema)
            if source is not None:
                params.append(Param(pname, source, param.default))
            else:
                service_candidates.append(ServiceCandidate(pname, hint, param.default))

        return cls(
            name=name,
            node_name=node_name,
            fn=fn,
            event_types=tuple(event_types),
            previous_names=getattr(fn, "_previous_names", ()),
            is_async=inspect.iscoroutinefunction(fn),
            params=tuple(params),
            field_matchers=field_matchers,
            field_inject_params=field_inject_params,
            raises=getattr(fn, "_raises", ()),
            retry=getattr(fn, "_retry", None),
            invariants=getattr(fn, "_invariants", ()),
            service_candidates=tuple(service_candidates),
        )

    def matches(self, event: Event) -> bool:
        """Check whether *event* satisfies this handler's type + field matchers.

        Type-valued matchers use isinstance; str-valued matchers use equality.
        """
        if not isinstance(event, self.event_types):
            return False
        for fname, matcher, is_type in self.field_matchers:
            value = getattr(event, fname, None)
            if is_type:
                if not isinstance(value, matcher):  # type: ignore[arg-type]
                    return False
            elif value != matcher:
                return False
        return True


@dataclass(frozen=True)
class BoundHandler:
    """A handler with its parameters matched to the services of one graph.

    Only a bound handler becomes a node that runs the handler body.
    """

    meta: HandlerMeta
    # Each candidate that a service fills, in signature order. A
    # ``RunScoped`` service is its ``BoundFactory``, built per node call.
    services: tuple[Param, ...]

    @property
    def params(self) -> tuple[Param, ...]:
        """Each parameter that the node call fills once, for every event."""
        return (*self.meta.params, *self.services)

    def require_claimed(
        self,
        reducer_names: frozenset[str],
        *,
        received: str,
        entry: str,
        context_schema: type | None = None,
    ) -> None:
        """Raise if a handler declares a param the framework cannot inject.

        Every parameter must be claimed by exactly one source: the event
        itself (the first positional param), ``EventLog``/``RunnableConfig``/
        ``BaseStore`` framework injectables, a registered reducer name, a
        field matcher, or a service. An unclaimed param means the user
        intended an injection the framework has no way to provide — surface
        it at graph build time so the failure is colocated with the
        misconfiguration. With *reducer_names*, a typo warning comes first.
        *received* states what ``services=`` holds, so a near-miss name key
        shows. *entry* is the call that received ``services=``. A parameter
        that looks like the run context gets a hint on *context_schema*.
        """
        meta = self.meta
        bound = {p.name for p in self.services}
        missing = [c for c in meta.service_candidates if c.param not in bound]
        if missing:
            refuse_unclaimed(
                meta.label,
                missing,
                reducer_names=reducer_names,
                received=received,
                entry=entry,
                context_schema=context_schema,
                reader="handler",
            )


def refuse_unclaimed(
    label: str,
    missing: list[ServiceCandidate],
    *,
    reducer_names: Collection[str],
    received: str,
    entry: str,
    context_schema: type | None,
    reader: str,
) -> NoReturn:
    """Raise ``TypeError`` for the parameters of *label* that no input fills.

    With *reducer_names*, a typo warning comes first. *reader* is
    ``"handler"`` or ``"factory"``. *received* states what ``services=``
    holds, so a near-miss name key shows. *entry* is the call that received
    ``services=``. A hint follows for each parameter that looks like the run
    config, the run context, the log or the event.
    """
    unclaimed = [c.param for c in missing]
    if reducer_names:
        warn_user(
            f"{label} has parameter(s) {unclaimed} that don't match any "
            f"reducer. Available reducers: {sorted(reducer_names)}. Typo?",
        )
    wanted = ", ".join(
        f"{c.param!r} ({_annotation_label(c.annotation)})" for c in missing
    )
    raise TypeError(
        f"{label} declares parameter(s) {unclaimed} that the framework cannot "
        f"inject: {wanted}."
        f"{_framework_hint(missing, reader)}"
        f"{_context_hint(missing, context_schema)} {received} "
        f"For a service, add a typed field of that name to the services= "
        f"object of {entry}; for state, register a Reducer; otherwise remove "
        f"the parameter."
    )


def _framework_hint(missing: list[ServiceCandidate], reader: str) -> str:
    """A hint for each missing parameter that looks like the run config, the
    log or, for a factory, the event, and for each one with a default."""
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415

    hints: list[str] = []
    for c in missing:
        hint = c.annotation
        if c.param == "config" or RunnableConfig in typing.get_args(hint):
            hints.append(
                f" To read the run config, annotate {c.param!r} with exactly "
                f"RunnableConfig."
            )
        elif c.param == "log":
            hints.append(f" To read the EventLog, annotate {c.param!r} with EventLog.")
        elif reader == "factory" and isinstance(hint, type) and issubclass(hint, Event):
            hints.append(
                f" A factory cannot read the event {hint.__qualname__}. Read it "
                f"in the handler, and pass the value to the service there."
            )
        if c.default is not inspect.Parameter.empty:
            hints.append(
                f" A default does not make {c.param!r} optional. Bind a "
                f"constant in a closure or a functools.partial."
            )
    return "".join(hints)


def is_coercible_schema(schema: type) -> bool:
    """Whether LangGraph coerces a ``dict`` context to *schema*: a dataclass
    or a pydantic model."""
    from pydantic import BaseModel  # noqa: PLC0415

    return is_dataclass(schema) or issubclass(schema, BaseModel)


def _context_hint(missing: list[ServiceCandidate], context_schema: type | None) -> str:
    """A hint for each missing parameter that looks like the run context.

    Without *context_schema*, a dataclass, a pydantic model or a
    ``TypedDict`` annotation can be a context schema. These are the schema
    kinds of LangGraph. With *context_schema*, a parameter is a near miss when
    its annotation is a subclass of the schema, or a union that holds it.
    """
    hints: list[str] = []
    for c in missing:
        hint = c.annotation
        if isinstance(hint, type) and issubclass(hint, Event):
            continue
        if context_schema is None:
            if isinstance(hint, type) and (
                is_coercible_schema(hint) or typing.is_typeddict(hint)
            ):
                hints.append(
                    f" If {c.param!r} is the run context, pass "
                    f"context_schema={hint.__qualname__}."
                )
            continue
        subclass = (
            isinstance(hint, type)
            and not typing.is_typeddict(context_schema)
            and issubclass(hint, context_schema)
        )
        near = subclass or context_schema in typing.get_args(hint)
        if near:
            hints.append(
                f" {c.param!r} is annotated {_annotation_label(hint)}, but only "
                f"the exact type context_schema={context_schema.__qualname__} "
                f"receives the run context."
            )
    return "".join(hints)


def _annotation_label(annotation: object) -> str:
    """Render the annotation of an unclaimed parameter for an error."""
    if isinstance(annotation, type):
        return annotation.__qualname__
    return repr(annotation)
