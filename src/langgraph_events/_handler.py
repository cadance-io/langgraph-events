"""The ``@on`` decorator and handler metadata extraction."""

from __future__ import annotations

import asyncio
import inspect
import typing
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import Event, Invariant
from langgraph_events._event_log import (
    EventLog,
)

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


def _resolve_type_hints(fn: Any) -> dict[str, Any]:
    """Return ``fn``'s resolved type hints, cached on the function object.

    Both ``_infer_event_type`` (at decoration) and ``extract_handler_meta``
    (at graph construction) need the same hints — resolving twice is wasteful
    and doubles the chance of forward-ref surprises. Cache on ``fn`` itself.
    """
    cached = getattr(fn, "_resolved_hints", None)
    if cached is not None:
        return cached
    hints = typing.get_type_hints(fn)
    fn._resolved_hints = hints
    return hints


def _infer_event_type(fn: Any) -> type[Event]:
    """Read an ``Event`` subclass off ``fn``'s first parameter annotation.

    Used by ``@on`` when positional event types are omitted. Raises
    ``TypeError`` with an actionable message for the full range of failure
    modes (missing parameter, missing annotation, non-``Event`` type, Union).
    """
    try:
        hints = _resolve_type_hints(fn)
    except Exception as exc:
        raise TypeError(
            f"@on could not resolve type hints for {fn.__qualname__!r}: "
            f"{exc}. Annotate the first parameter with a resolvable Event "
            f"subclass, or pass the event type explicitly: @on(EventType)."
        ) from exc

    sig = inspect.signature(fn)
    params = [p for p in sig.parameters if p != "self"]
    if not params:
        raise TypeError(
            f"@on requires {fn.__qualname__!r} to declare a typed first "
            f"parameter (the event), but it has none."
        )
    first = params[0]
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


def _build_on_decorator(
    event_types: tuple[type[Event], ...],
    raises: type[Exception] | tuple[type[Exception], ...],
    invariants: dict[type[Invariant], Callable[..., bool]] | None,
    field_matchers: dict[str, type[Event] | type[Exception] | type[Invariant] | str],
) -> Callable[[F], F]:
    """Validate arguments and return the decorator that stamps attributes."""
    for et in event_types:
        if not (
            isinstance(et, type)
            and (issubclass(et, Event) or getattr(et, "_event_mixin", False))
        ):
            raise TypeError(f"@on() requires Event subclasses or mixins, got {et!r}")

    # Normalise and validate raises
    raises_tuple: tuple[type[Exception], ...] = (
        raises if isinstance(raises, tuple) else (raises,)
    )
    for rt in raises_tuple:
        if not (isinstance(rt, type) and issubclass(rt, Exception)):
            raise TypeError(
                f"@on() raises= entries must be Exception subclasses, got {rt!r}. "
                f"Non-Exception BaseException subclasses (KeyboardInterrupt, "
                f"SystemExit, GeneratorExit, asyncio.CancelledError) are not "
                f"allowed — they are runtime/exit signals, not domain errors."
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

    def decorator(fn: F) -> F:
        fn._event_types = event_types  # type: ignore[attr-defined]
        if field_matchers:
            fn._field_matchers = dict(field_matchers)  # type: ignore[attr-defined]
        if raises_tuple:
            fn._raises = raises_tuple  # type: ignore[attr-defined]
        if invariants_tuple:
            fn._invariants = invariants_tuple  # type: ignore[attr-defined]
        return fn

    return decorator


def on(
    *event_types: Any,
    raises: type[Exception] | tuple[type[Exception], ...] = (),
    invariants: dict[type[Invariant], Callable[..., bool]] | None = None,
    **field_matchers: type[Event] | type[Exception] | type[Invariant] | str,
) -> Any:
    """Subscribe a handler to one or more event types.

    Three shapes, escalating by what's needed:

    1. **Bare** — ``@on`` (no parens). Infers the event type from the
       handler's first parameter annotation::

           @on
           def place(event: Order.Place) -> Order.Place.Placed:
               return Order.Place.Placed(order_id="o1")

    2. **Modifiers only** — ``@on(raises=..., invariants=..., field=...)``.
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

    Field matchers narrow dispatch — ``@on(Resumed, interrupted=Approval)``
    for ``isinstance`` match (works for Event, Exception, or Invariant
    subclasses); string values do equality match (e.g. a string event field).
    """
    no_modifiers = raises == () and invariants is None and not field_matchers
    sole_arg_is_function = len(event_types) == 1 and (
        inspect.isfunction(event_types[0]) or inspect.ismethod(event_types[0])
    )

    if sole_arg_is_function and no_modifiers:
        fn = event_types[0]
        return _build_on_decorator((_infer_event_type(fn),), (), None, {})(fn)

    if not event_types:

        def inferring(fn: F) -> F:
            return _build_on_decorator(
                (_infer_event_type(fn),), raises, invariants, dict(field_matchers)
            )(fn)

        return inferring

    return _build_on_decorator(event_types, raises, invariants, dict(field_matchers))


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler."""

    name: str
    fn: Callable[..., HandlerReturn]
    event_types: tuple[type[Event], ...]
    log_param: str | None
    is_async: bool
    reducer_params: tuple[str, ...] = ()
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
    invariants: tuple[tuple[type[Invariant], Callable[..., bool]], ...] = ()

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


def extract_handler_meta(
    fn: Callable[..., Any],
    reducer_names: frozenset[str] = frozenset(),
) -> HandlerMeta:
    """Extract handler metadata from a decorated function."""
    event_types = getattr(fn, "_event_types", None)
    if event_types is None:
        raise ValueError(
            f"Function {fn.__qualname__!r} is not decorated with @on(EventType)"
        )

    try:
        hints = _resolve_type_hints(fn)
    except Exception as exc:
        warnings.warn(
            f"Failed to resolve type hints for handler {fn.__qualname__!r}; "
            f"falling back to signature-only detection. ({exc})",
            stacklevel=3,
        )
        hints = {}

    # Find the actual parameter name annotated with EventLog
    log_param: str | None = None
    for param_name, hint in hints.items():
        if hint is EventLog:
            log_param = param_name
            break

    # Detect config and store parameters by type hint
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415
    from langgraph.store.base import BaseStore  # noqa: PLC0415

    config_param: str | None = None
    store_param: str | None = None
    for param_name, hint in hints.items():
        if hint is RunnableConfig:
            config_param = param_name
        elif hint is BaseStore:
            store_param = param_name

    # Detect reducer parameters by name match
    sig = inspect.signature(fn)
    reducer_params = tuple(name for name in sig.parameters if name in reducer_names)

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

    # Extract declared invariants
    invariants: tuple[tuple[type[Invariant], Callable[..., bool]], ...] = getattr(
        fn, "_invariants", ()
    )

    # Warn about handler params that don't match any known injection source
    if reducer_names:
        first_param = next(iter(sig.parameters), None)
        known_params = {first_param}  # first param is always the event
        if log_param:
            known_params.add(log_param)
        if config_param:
            known_params.add(config_param)
        if store_param:
            known_params.add(store_param)
        known_params.update(reducer_names)
        known_params.update(raw_field_matchers.keys())
        unknown = [
            name
            for name in sig.parameters
            if name not in known_params and name != "self"
        ]
        if unknown:
            warnings.warn(
                f"Handler {fn.__qualname__!r} has parameter(s) {unknown} that "
                f"don't match any reducer. "
                f"Available reducers: {sorted(reducer_names)}. Typo?",
                stacklevel=3,
            )

    return HandlerMeta(
        name=fn.__name__,
        fn=fn,
        event_types=tuple(event_types),
        log_param=log_param,
        is_async=asyncio.iscoroutinefunction(fn),
        reducer_params=reducer_params,
        config_param=config_param,
        store_param=store_param,
        field_matchers=field_matchers,
        field_inject_params=field_inject_params,
        raises=raises,
        invariants=invariants,
    )
