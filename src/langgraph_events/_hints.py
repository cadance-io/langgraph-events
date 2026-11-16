"""Resolve annotations, and test whether one annotation fits another."""

from __future__ import annotations

import contextlib
import functools
import inspect
import types
import typing
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable


def _annotation_accepts_none(hint: object) -> bool:
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


def _resolve_each_annotation(
    fn: Callable[..., object],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve ``fn``'s annotations one at a time.

    ``typing.get_type_hints`` is all-or-nothing. One unresolvable annotation
    discards every hint on the handler, so a valid ``RunnableConfig`` param
    stops being detected and the resulting error names the wrong parameter.
    See issue #183.

    Resolve each annotation on its own probe function instead. A resolvable
    annotation lands in the hints. An unresolvable one lands in the errors,
    keyed by parameter name.

    Resolution uses the globals of the innermost wrapped function, exactly
    like the whole-function call, so the fallback never resolves a name the
    fast path would miss. A ``functools.wraps`` decorator copies the
    annotations but not the globals, so the wrapper's own module is wrong.
    """
    target = getattr(fn, "__func__", fn)
    raw = getattr(target, "__annotations__", {})
    globalns = getattr(inspect.unwrap(target), "__globals__", {})
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


def _resolve_hints_and_errors(
    fn: Callable[..., object],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``fn``'s resolved hints and its per-annotation failures.

    Both ``_infer_event_type`` (at decoration) and ``HandlerMeta.of``
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
    target: Any = getattr(fn, "__func__", fn)
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


def _resolve_parameter_hints(
    fn: Callable[..., object],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve the annotations of the callable *fn*, as for
    :func:`_resolve_hints_and_errors`.

    The annotations live on the object that :func:`_annotation_source`
    gives: the wrapped function of a ``functools.partial``, the
    ``__init__`` of a class, or the ``__call__`` of a callable object.
    """
    source = _annotation_source(fn)
    init = source.__init__ if isinstance(source, type) else source  # type: ignore[misc]
    return _resolve_hints_and_errors(init)


def _hint_label(hint: object) -> str:
    """Render an annotation for an error message.

    A generic alias such as ``list[str]`` forwards ``__qualname__`` to its
    origin, so only a plain class uses it.
    """
    if hint is type(None):
        return "None"
    if isinstance(hint, type):
        return hint.__qualname__
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(_hint_label(a) for a in typing.get_args(hint))
    return repr(hint).replace("typing.", "")


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


def _promoted(hint: object) -> object:
    """*hint* with the numeric tower of PEP 484: ``float`` also accepts
    ``int``, and ``complex`` also accepts ``float`` and ``int``.

    The promotion applies inside a union and inside type arguments. A hint
    that Python cannot rebuild stays as is.
    """
    if hint is float:
        return float | int
    if hint is complex:
        return complex | float | int
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is None or not args:
        return hint
    promoted = tuple(_promoted(a) for a in args)
    if origin is typing.Union or origin is types.UnionType:
        return typing.Union[promoted]  # noqa: UP007 - a runtime value
    try:
        return origin[promoted]
    except TypeError:
        return hint


def _hint_fits(provided: object, declared: object) -> bool | str:
    """Whether a value of the type *provided* can fill a parameter annotated
    *declared*. A string is the reason that the check cannot run.

    ``Any`` and a bare type variable say nothing, so they fit. The numeric
    tower applies, so ``int`` fits ``float``. A *provided* type with no
    type arguments, or with ``Any`` as one, does not state its arguments.
    Its class is then compared with the class of *declared*.
    """
    if provided is Any or isinstance(provided, TypeVar):
        return True
    from beartype.door import is_subhint  # noqa: PLC0415

    declared = _promoted(declared)
    verdict = _satisfies(is_subhint, provided, declared)
    unstated = _erased(provided) is provided or Any in typing.get_args(provided)
    if verdict is False and unstated:
        return _satisfies(is_subhint, _erased(provided), _erased(declared))
    return verdict


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
