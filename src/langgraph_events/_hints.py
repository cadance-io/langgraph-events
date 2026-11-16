"""Resolve annotations, and test whether one annotation fits another."""

from __future__ import annotations

import contextlib
import functools
import inspect
import types
import typing
from collections import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

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


def _open_parameters(fn: Callable[..., object]) -> tuple[inspect.Parameter, ...]:
    """The parameters that a call of *fn* must fill.

    A keyword that a ``functools.partial`` binds is filled already, so it is
    left out. Raises ``TypeError`` or ``ValueError`` when Python cannot read
    the signature.
    """
    bound: set[str] = set()
    target: object = fn
    while isinstance(target, functools.partial):
        bound.update(target.keywords)
        target = target.func
    params = inspect.signature(fn).parameters.values()
    return tuple(p for p in params if p.name not in bound)


def _resolve_parameter_hints(
    fn: Callable[..., object],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve the annotations of the callable *fn*, as for
    :func:`_resolve_hints_and_errors`.

    The annotations live on the object that :func:`_annotation_source`
    gives: the wrapped function of a ``functools.partial``, the
    ``__init__`` of a class, or the ``__call__`` of a callable object. A
    class gives itself as the return annotation, because a call of it gives
    an instance.
    """
    source = _annotation_source(fn)
    if not isinstance(source, type):
        return _resolve_hints_and_errors(source)
    hints, errors = _resolve_hints_and_errors(source.__init__)  # type: ignore[misc]
    errors = {name: e for name, e in errors.items() if name != "return"}
    return {**hints, "return": source}, errors


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


@dataclass(frozen=True)
class Fits:
    """The provided type fits the annotation."""


@dataclass(frozen=True)
class Misfit:
    """The provided type does not fit the annotation."""


@dataclass(frozen=True)
class Untestable:
    """Python cannot test the annotation at run time, so no check runs."""

    reason: str


Verdict: TypeAlias = Fits | Misfit | Untestable


_INVARIANT: tuple[type, ...] = (
    list,
    dict,
    set,
    abc.MutableSequence,
    abc.MutableMapping,
    abc.MutableSet,
)
"""The containers whose type arguments are invariant, as for mypy."""


def _union_members(hint: object) -> tuple[object, ...]:
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return typing.get_args(hint)
    return (hint,)


def _promoted(hint: object) -> object:
    """*hint* with the numeric tower of PEP 484 at the top level only:
    ``float`` also accepts ``int``, and ``complex`` also accepts ``float``
    and ``int``. A type argument is not promoted, so ``list[float]`` does
    not accept ``list[int]``."""
    tower: dict[object, tuple[object, ...]] = {
        float: (float, int),
        complex: (complex, float, int),
    }
    members = [m for h in _union_members(hint) for m in tower.get(h, (h,))]
    if len(members) == len(_union_members(hint)):
        return hint
    return typing.Union[tuple(dict.fromkeys(members))]  # noqa: UP007 - a runtime value


def _same_arguments(provided: tuple[object, ...], declared: tuple[object, ...]) -> bool:
    """Whether two argument lists are equal. ``Any`` matches any argument."""
    return len(provided) == len(declared) and all(
        p == d or p is Any or d is Any for p, d in zip(provided, declared, strict=True)
    )


def _variance_holds(provided: object, declared: object) -> bool:
    """Whether the type arguments of *provided* fit *declared* as mypy
    requires.

    An invariant container needs equal type arguments. The key of a
    ``Mapping`` is invariant too, as in typeshed. Any other generic whose
    class is a subclass of the declared class, with as many type arguments,
    is checked argument by argument. beartype treats every container as
    covariant, so it accepts ``list[bool]`` for ``list[int]``. mypy refuses
    it.
    """
    for p in _union_members(provided):
        p_origin, p_args = typing.get_origin(p), typing.get_args(p)
        if not isinstance(p_origin, type) or not p_args:
            continue
        matches: list[bool] = []
        for d in _union_members(declared):
            d_origin, d_args = typing.get_origin(d), typing.get_args(d)
            if not isinstance(d_origin, type) or not d_args:
                continue
            if not issubclass(p_origin, d_origin) or len(p_args) != len(d_args):
                continue
            if issubclass(d_origin, _INVARIANT):
                matches.append(_same_arguments(p_args, d_args))
                continue
            key_holds = not issubclass(d_origin, abc.Mapping) or _same_arguments(
                p_args[:1], d_args[:1]
            )
            matches.append(
                key_holds
                and all(
                    _variance_holds(pa, da)
                    for pa, da in zip(p_args, d_args, strict=True)
                )
            )
        if matches and not any(matches):
            return False
    return True


def _stated(provided: object, declared: object) -> object:  # noqa: PLR0911 - one return per shape of type
    """*provided*, with each unknown type argument taken from *declared*.

    A type argument ``Any`` is unknown. A generic class with no type
    arguments, such as a bare ``dict``, has unknown arguments too. Each
    known argument stays, so ``dict[str, Any]`` still does not fit
    ``dict[int, str]``.
    """
    if provided is Any:
        return declared
    members = _union_members(provided)
    if len(members) > 1:
        return typing.Union[tuple(_stated(m, declared) for m in members)]  # noqa: UP007 - a runtime value
    p_origin, p_args = typing.get_origin(provided), typing.get_args(provided)
    for d in _union_members(declared):
        d_origin, d_args = typing.get_origin(d), typing.get_args(d)
        if not isinstance(d_origin, type) or not d_args:
            continue
        if p_origin is None and isinstance(provided, type):
            if issubclass(provided, d_origin) and not typing.get_args(provided):
                try:
                    return provided[d_args]  # type: ignore[index]
                except TypeError:
                    return d
            continue
        if not isinstance(p_origin, type) or not issubclass(p_origin, d_origin):
            continue
        if len(p_args) != len(d_args):
            continue
        filled = tuple(_stated(pa, da) for pa, da in zip(p_args, d_args, strict=True))
        try:
            return p_origin[filled]  # type: ignore[index]
        except TypeError:
            return provided
    return provided


def _hint_fits(provided: object, declared: object) -> Verdict:
    """Whether a value of the type *provided* can fill a parameter annotated
    *declared*.

    ``Any`` and a bare type variable say nothing, so they fit. The numeric
    tower applies at the top level only, so ``int`` fits ``float``, but
    ``list[int]`` does not fit ``list[float]``. The variance follows mypy,
    see :func:`_variance_holds`. An unknown type argument, ``Any`` or the
    missing argument of a bare generic class, takes the declared argument
    at its position. A known argument is checked.
    """
    if provided is Any or isinstance(provided, TypeVar):
        return Fits()
    from beartype.door import is_subhint  # noqa: PLC0415

    declared = _promoted(declared)
    provided = _stated(provided, declared)
    verdict = _satisfies(is_subhint, provided, declared)
    if isinstance(verdict, Fits) and not _variance_holds(provided, declared):
        return Misfit()
    return verdict


def _satisfies(
    check: Callable[[Any, Any], bool], subject: object, declared: object
) -> Verdict:
    """Apply a ``beartype.door`` check.

    ``typing`` refuses a ``Protocol`` that is not ``runtime_checkable``, and
    beartype refuses a hint it cannot test at run time, at any nesting depth.
    Such an annotation stays a valid static contract, so the build must not
    fail on it. The caller warns instead.
    """
    from beartype.roar import BeartypeException  # noqa: PLC0415

    try:
        return Fits() if check(subject, declared) else Misfit()
    except (TypeError, BeartypeException) as exc:
        return Untestable(f"{type(exc).__name__}: {exc}")
