"""The ``@on`` decorator and handler metadata extraction."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import types
import typing
from collections import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import Event, Invariant
from langgraph_events._event_log import (
    EventLog,
)
from langgraph_events._identity import command_identity
from langgraph_events._retry import RetryPolicy
from langgraph_events._validate import normalize_exception_tuple
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Callable

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
        if asyncio.iscoroutinefunction(pred):
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


def _annotation_accepts_none(hint: Any) -> bool:
    """Return True if *hint* permits ``None`` as a value.

    A handler reducer-parameter whose annotation rejects ``None`` (e.g.
    ``strategy: str``) opts the parameter into the required-value assertion;
    ``str | None``, ``Optional[str]``, ``Any``, ``object``, and missing
    annotations all keep the legacy permissive behavior.
    """
    if hint is None or hint is type(None):
        return True
    if hint is typing.Any:
        return True
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return any(_annotation_accepts_none(a) for a in typing.get_args(hint))
    # ``object`` (and any other class that is ``None``'s supertype) accepts
    # None. For parameterised generics (``list[str]``), ``isinstance`` raises
    # ``TypeError`` — conservatively treat them as rejecting None.
    if isinstance(hint, type):
        try:
            return isinstance(None, hint)
        except TypeError:
            return False
    return False


def _resolve_each_annotation(fn: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve ``fn``'s annotations one at a time.

    ``typing.get_type_hints`` is all-or-nothing. One unresolvable annotation
    discards every hint on the handler, so a valid ``RunnableConfig`` param
    stops being detected and the resulting error names the wrong parameter.
    See issue #183.

    Resolve each annotation on its own probe function instead. A resolvable
    annotation lands in the hints. An unresolvable one lands in the errors,
    keyed by parameter name.

    Resolution uses ``fn.__globals__`` only, exactly like the whole-function
    call, so the fallback never resolves a name the fast path would miss.
    """
    target = getattr(fn, "__func__", fn)
    raw = getattr(target, "__annotations__", {})
    globalns = getattr(target, "__globals__", {})
    # ``get_type_hints`` puts a PEP 695 type parameter in scope for the
    # function that declares it. The probe declares none, so pass them as
    # locals or every annotation naming one is recorded as a failure.
    localns = {p.__name__: p for p in getattr(target, "__type_params__", ())}
    hints: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _probe() -> None: ...

    for name, annotation in raw.items():
        _probe.__annotations__ = {name: annotation}
        try:
            hints[name] = typing.get_type_hints(_probe, globalns, localns)[name]
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return hints, errors


def _resolve_hints_and_errors(fn: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``fn``'s resolved hints and its per-annotation failures.

    Both ``_infer_event_type`` (at decoration) and ``extract_handler_meta``
    (at graph construction) need the same hints — resolving twice is wasteful
    and doubles the chance of forward-ref surprises. Cache on ``fn`` itself.

    Never raises. A caller that needs an annotation reads ``errors`` to learn
    why it is absent, and decides whether that absence is fatal.

    Only a complete resolution is cached. A name can be absent at decoration
    and present at graph build — a service class declared below its handler,
    for instance — so a cached failure would report a resolvable annotation as
    broken forever. Retry instead. The retry costs nothing on the happy path.

    The cache lives on the underlying function, so a bound method caches once
    for every instance. A callable that rejects attributes is not cached.
    """
    target = getattr(fn, "__func__", fn)
    cached = getattr(target, "_resolved_hints", None)
    if cached is not None:
        return cached, {}
    try:
        hints = typing.get_type_hints(fn)
        errors: dict[str, str] = {}
    except Exception:
        hints, errors = _resolve_each_annotation(fn)
    if not errors:
        with contextlib.suppress(AttributeError):
            target._resolved_hints = hints
    return hints, errors


def _resolve_type_hints(fn: Any) -> dict[str, Any]:
    """Return ``fn``'s resolvable type hints. Unresolvable ones are absent."""
    return _resolve_hints_and_errors(fn)[0]


def _infer_event_type(fn: Any) -> type[Event]:
    """Read an ``Event`` subclass off ``fn``'s first parameter annotation.

    Used by ``@on`` when positional event types are omitted. Raises
    ``TypeError`` with an actionable message for the full range of failure
    modes (missing parameter, missing annotation, non-``Event`` type, Union).
    """
    hints, hint_errors = _resolve_hints_and_errors(fn)
    sig = inspect.signature(fn)
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


def normalize_previous_names(previously: Any, *, owner: str) -> tuple[str, ...]:
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
        # ``extract_handler_meta`` reads them with, so an empty stamp is
        # indistinguishable from no stamp for every consumer.
        fn._raises = raises_tuple  # type: ignore[attr-defined]
        fn._invariants = invariants_tuple  # type: ignore[attr-defined]
        fn._previous_names = previous_names  # type: ignore[attr-defined]
        fn._retry = retry  # type: ignore[attr-defined]
        # ``node_name`` has no class-level declaration to re-stamp from (see
        # ``extract_handler_meta``), so there is no stale value to correct and
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
) -> Any:
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
    the next backoff would cross the run's ``deadline=``, in which case it
    carries ``abandoned_for_deadline=True``. Requires a non-empty ``raises=``;
    the policy's ``on=`` must overlap it.

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


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler."""

    name: str
    fn: Callable[..., HandlerReturn]
    event_types: tuple[type[Event], ...]
    log_param: str | None
    is_async: bool
    # Stable graph-node / checkpoint identity. Equals ``name`` for ordinary
    # handlers, but for inline ``Command.handle()`` handlers it is the command's
    # ``__qualname__`` (e.g. ``Order.Place``) so the node a paused checkpoint
    # resumes into never depends on ``handlers=[...]`` order. ``name`` stays the
    # human-readable handler label used in choreography/mermaid/diagnostics.
    # Required — always populated by ``extract_handler_meta``; never empty.
    node_name: str
    # Historic node names this handler answered to, declared via
    # ``@on(previously=...)``. The graph registers an alias node per name so an
    # interrupted checkpoint paused at the old node still resumes after a rename.
    previous_names: tuple[str, ...] = ()
    # Parameter annotated with ``Reflection`` — injected as an enriched,
    # mid-dispatch snapshot (log-so-far + namespace model + reducers).
    reflection_param: str | None = None
    reducer_params: tuple[str, ...] = ()
    # Subset of ``reducer_params`` whose annotation rejects ``None`` — the
    # framework raises ``ReducerNotSetError`` if the channel value is ``None``
    # at injection. Computed once at extraction; consulted in _build_inject.
    required_reducer_params: frozenset[str] = frozenset()
    config_param: str | None = None
    store_param: str | None = None
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
    # (param_name, registered_service_type) for params whose annotation is a
    # base class of (or identical to) a service type registered on
    # EventGraph(services=[...]). Used as the lookup key in the type-keyed
    # services map at dispatch time.
    service_params: tuple[tuple[str, type], ...] = ()
    # (param_name, registered_service_name) for params whose name matches a
    # key in EventGraph(services={...}). Used as the lookup key in the
    # name-keyed services map at dispatch time.
    service_name_params: tuple[tuple[str, str], ...] = ()
    # Parameters whose annotation did not resolve, mapped to the reason. A
    # parameter listed here carries no hint, so no type-matched injection can
    # claim it. ``_verify_no_unclaimed_params`` reads this to name the real
    # cause instead of blaming an unrelated parameter. See issue #183.
    hint_errors: tuple[tuple[str, str], ...] = ()

    @property
    def claimant(self) -> str:
        """The name to blame in a construction-time error message.

        An inline ``Command`` handler is named by its command identity
        (``Order.Place``), not by the method name — ``handle``, or the
        positional dedup suffix ``handle_2``, does not help the user find it.
        """
        if getattr(self.fn, "_inline_command", None) is not None:
            return self.node_name
        return self.name

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

    @property
    def wants_log(self) -> bool:
        """Backward-compatible property."""
        return self.log_param is not None

    @property
    def framework_params(self) -> tuple[str, ...]:
        """Param names claimed by framework injectables (log/reflection/config/store).

        The one place that knows the full injectable set — claim/consume
        bookkeeping iterates this instead of hand-listing each field.
        """
        return tuple(
            p
            for p in (
                self.log_param,
                self.reflection_param,
                self.config_param,
                self.store_param,
            )
            if p
        )


def _warn_on_unknown_reducer_params(
    fn: Callable[..., Any],
    sig: inspect.Signature,
    *,
    reducer_names: frozenset[str],
    consumed: set[str | None],
) -> None:
    """Surface a typo warning when reducer-style param names go unmatched."""
    unknown = [
        name for name in sig.parameters if name not in consumed and name != "self"
    ]
    if not unknown:
        return
    warn_user(
        f"Handler {fn.__qualname__!r} has parameter(s) {unknown} that "
        f"don't match any reducer. "
        f"Available reducers: {sorted(reducer_names)}. Typo?",
    )


def _detect_service_name_params(
    sig: inspect.Signature,
    service_names: frozenset[str],
    consumed: set[str | None],
) -> tuple[tuple[str, str], ...]:
    """Match unclaimed handler params against the name-keyed services map.

    A param whose name appears as a key in ``EventGraph(services={...})`` is
    bound to the corresponding instance at dispatch. Annotation-free params
    are eligible — name-keyed binding does not require a type hint.
    """
    if not service_names:
        return ()
    return tuple(
        (param_name, param_name)
        for param_name in sig.parameters
        if param_name not in consumed and param_name in service_names
    )


def _detect_service_params(
    fn: Callable[..., Any],
    sig: inspect.Signature,
    hints: dict[str, Any],
    service_types: frozenset[type],
    consumed: set[str | None],
) -> tuple[tuple[str, type], ...]:
    """Match unclaimed handler params against registered service types.

    For each param whose annotation is a class, the matching service type is
    the unique registered type that is a subclass of (or equal to) the
    annotation. Multi-match is rejected — the user must disambiguate by
    narrowing the annotation or registering only one matching service.
    """
    if not service_types:
        return ()
    detected: list[tuple[str, type]] = []
    for param_name in sig.parameters:
        if param_name in consumed:
            continue
        hint = hints.get(param_name)
        if not isinstance(hint, type):
            continue
        # ``object`` matches every registered type; treating it as a service
        # claim would silently consume an unrelated instance. Skip — the
        # caller meant "untyped" and ``_verify_no_unclaimed_params`` will
        # surface the missing-source error at graph build.
        if hint is object:
            continue
        # Exact-type match wins over subclass-only match — it lets users
        # register a base + subclass pair (e.g. ``[BaseChatModel(), Anthropic()]``)
        # and resolve `param: BaseChatModel` to the base instance unambiguously.
        if hint in service_types:
            detected.append((param_name, hint))
            continue
        matches = [t for t in service_types if issubclass(t, hint)]
        if len(matches) > 1:
            names = ", ".join(sorted(t.__name__ for t in matches))
            raise TypeError(
                f"Handler {fn.__qualname__!r} parameter {param_name!r} "
                f"annotated as {hint.__name__!r} matches multiple "
                f"registered services ({names}). Disambiguate by "
                f"narrowing the annotation, or register only one "
                f"service that satisfies it."
            )
        if matches:
            detected.append((param_name, matches[0]))
    return tuple(detected)


def _detect_framework_params(hints: dict[str, Any]) -> dict[str, str | None]:
    """Find params annotated with framework injectables, by exact type hint.

    Returns a dict keyed by the ``HandlerMeta`` field names
    (``log_param`` / ``reflection_param`` / ``config_param`` / ``store_param``)
    so call sites read each key by name — no positional slots to miswire.
    First matching param wins for each injectable.
    """
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415
    from langgraph.store.base import BaseStore  # noqa: PLC0415

    from langgraph_events._reflection import Reflection  # noqa: PLC0415

    hint_to_field = {
        EventLog: "log_param",
        Reflection: "reflection_param",
        RunnableConfig: "config_param",
        BaseStore: "store_param",
    }
    detected: dict[str, str | None] = dict.fromkeys(hint_to_field.values())
    for param_name, hint in hints.items():
        field = hint_to_field.get(hint)
        if field is not None and detected[field] is None:
            detected[field] = param_name
    return detected


def extract_handler_meta(
    fn: Callable[..., Any],
    reducer_names: frozenset[str] = frozenset(),
    service_types: frozenset[type] = frozenset(),
    service_names: frozenset[str] = frozenset(),
) -> HandlerMeta:
    """Extract handler metadata from a decorated function.

    ``service_types`` is the set of exact types registered on
    ``EventGraph(services=[...])`` (sequence form). Each handler param whose
    annotation is a base class of (or identical to) a registered service
    type is recorded as a service param.

    ``service_names`` is the set of keys in ``EventGraph(services={...})``
    (mapping form). Each handler param whose name matches a key is recorded
    as a name-keyed service param.

    Only one of the two forms is populated per ``EventGraph`` — they are
    mutually exclusive.
    """
    event_types = getattr(fn, "_event_types", None)
    if event_types is None:
        raise ValueError(
            f"Function {fn.__qualname__!r} is not decorated with @on(EventType)"
        )

    hints, hint_errors = _resolve_hints_and_errors(fn)
    param_errors = {n: e for n, e in hint_errors.items() if n != "return"}
    if param_errors:
        failed = ", ".join(f"{n!r} ({e})" for n, e in param_errors.items())
        warn_user(
            f"Failed to resolve the annotation on parameter(s) {failed} of "
            f"handler {fn.__qualname__!r}; the framework cannot match them by "
            f"type. Make each annotation importable at run time.",
        )

    framework_params = _detect_framework_params(hints)

    # Detect reducer parameters by name match
    sig = inspect.signature(fn)
    reducer_params = tuple(name for name in sig.parameters if name in reducer_names)
    required_reducer_params = frozenset(
        name
        for name in reducer_params
        if name in hints and not _annotation_accepts_none(hints[name])
    )

    # Extract field matchers; classify each now so dispatch avoids isinstance.
    raw_field_matchers: dict[
        str, type[Event] | type[Exception] | type[Invariant] | str
    ] = getattr(fn, "_field_matchers", {})
    field_matchers = tuple(
        (name, matcher, isinstance(matcher, type))
        for name, matcher in raw_field_matchers.items()
    )
    field_inject_params = frozenset(
        name for name in sig.parameters if name in raw_field_matchers
    )

    # Extract declared raises
    raises: tuple[type[Exception], ...] = getattr(fn, "_raises", ())

    # Extract the declared retry policy
    retry: RetryPolicy | None = getattr(fn, "_retry", None)

    # Extract declared invariants
    invariants: tuple[tuple[type[Invariant], Callable[..., bool]], ...] = getattr(
        fn, "_invariants", ()
    )

    # Detect service params by type. Resolution order: anything claimed by
    # log/config/store/reducer/field-matcher or the first (event) param is
    # already consumed; the rest are candidates for service injection.
    first_param = next(iter(sig.parameters), None)
    consumed_for_services: set[str | None] = {first_param, "self"}
    consumed_for_services.update(p for p in framework_params.values() if p)
    consumed_for_services.update(reducer_params)
    consumed_for_services.update(raw_field_matchers.keys())
    service_params = _detect_service_params(
        fn, sig, hints, service_types, consumed_for_services
    )
    service_name_params = _detect_service_name_params(
        sig, service_names, consumed_for_services
    )

    if reducer_names:
        _warn_on_unknown_reducer_params(
            fn,
            sig,
            reducer_names=reducer_names,
            consumed=(
                consumed_for_services
                | {n for n, _ in service_params}
                | {n for n, _ in service_name_params}
            ),
        )

    name = getattr(fn, "_node_name", None) or fn.__name__
    # Inline command handlers route by their command's qualname (stable,
    # order-independent); every other handler routes by its own name. The
    # qualname intentionally wins over any ``_node_name`` here — inline
    # handlers never carry one today (``_expand_command_handlers`` forwards
    # raises/invariants/previously but no ``node_name``), but were that to
    # change, command identity must stay the resume identity.
    inline_command = getattr(fn, "_inline_command", None)
    node_name = command_identity(inline_command) if inline_command is not None else name

    return HandlerMeta(
        name=name,
        node_name=node_name,
        fn=fn,
        event_types=tuple(event_types),
        previous_names=getattr(fn, "_previous_names", ()),
        log_param=framework_params["log_param"],
        reflection_param=framework_params["reflection_param"],
        config_param=framework_params["config_param"],
        store_param=framework_params["store_param"],
        is_async=asyncio.iscoroutinefunction(fn),
        reducer_params=reducer_params,
        required_reducer_params=required_reducer_params,
        field_matchers=field_matchers,
        field_inject_params=field_inject_params,
        raises=raises,
        retry=retry,
        invariants=invariants,
        service_params=service_params,
        service_name_params=service_name_params,
        hint_errors=tuple(param_errors.items()),
    )
