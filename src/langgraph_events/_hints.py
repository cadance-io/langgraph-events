"""Resolve annotations, and test whether two annotations name the same type."""

from __future__ import annotations

import contextlib
import functools
import inspect
import types
import typing
from typing import TYPE_CHECKING, Any, Literal, TypeVar

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


class UnsettledTypeError(TypeError):
    """A type that names no class yet: a type variable or a string."""


def _normal(hint: object) -> tuple[Any, ...]:  # noqa: PLR0911 - one return per row of the rule table
    """The canonical form of *hint*, for an exact comparison.

    Two annotations name the same type when their canonical forms are equal.
    The rule table:

    - ``Annotated`` metadata is stripped, at any depth.
    - ``None`` is ``NoneType``. ``Optional[X]`` is ``X | None``.
    - A union is the set of its members, so the order does not count. The
      ``Literal`` members of a union merge into one ``Literal``.
    - A ``Literal`` is the set of its values, keyed by type and value, so
      ``Literal[1]`` is not ``Literal[True]``.
    - A generic alias is its origin class and the canonical form of each
      argument, a ``Callable`` argument list included. So ``List[int]`` is
      ``list[int]``, and ``typing.Sequence[int]`` is
      ``collections.abc.Sequence[int]``. A bare alias such as ``List`` is
      its origin class.
    - ``Any`` equals only ``Any``. A ``NewType`` is its own type.
    - A type alias statement without parameters is its value.

    Raises :class:`UnsettledTypeError` for a type variable, a forward reference,
    a string or a type alias with parameters.
    """
    alias = getattr(typing, "TypeAliasType", None)
    if alias is not None and isinstance(hint, alias):
        if hint.__type_params__:
            raise UnsettledTypeError(
                f"the type alias {hint.__name__} has type parameters"
            )
        return _normal(hint.__value__)
    if isinstance(hint, (TypeVar, typing.ForwardRef, str)):
        raise UnsettledTypeError(f"{hint!r} names no class yet")
    if hint is None:
        return ("class", type(None))
    if hint is Any:
        return ("any",)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Annotated:
        return _normal(args[0])
    if origin is Literal:
        return ("literal", frozenset((type(v), v) for v in args))
    if origin is typing.Union or origin is types.UnionType:
        members = {_normal(a) for a in args}
        literals = [m for m in members if m[0] == "literal"]
        if len(literals) > 1:
            members -= set(literals)
            members.add(("literal", frozenset().union(*(m[1] for m in literals))))
        return (
            next(iter(members)) if len(members) == 1 else ("union", frozenset(members))
        )
    if origin is not None and not args:
        return ("class", origin)
    if origin is not None:
        return ("generic", origin, tuple(_normal_argument(a) for a in args))
    return ("class", hint)


def _normal_argument(arg: object) -> tuple[Any, ...]:
    """The canonical form of one type argument. A ``Callable`` argument list
    is a list, and ``...`` stays as is."""
    if isinstance(arg, list):
        return ("arguments", tuple(_normal(a) for a in arg))
    if arg is Ellipsis:
        return ("ellipsis",)
    return _normal(arg)


def _same_type(reader: object, source: object, *, may_be_unset: bool) -> bool:
    """Whether the annotation *reader* names exactly the declared type
    *source*, by the rule table of :func:`_normal`.

    With *may_be_unset*, a reader that leaves ``None`` out of an optional
    source fits too, because an unset value raises ``ReducerNotSetError``.
    """
    wanted = _normal(source)
    named = _normal(reader)
    if named == wanted:
        return True
    if not may_be_unset or wanted[0] != "union":
        return False
    without_none = wanted[1] - {_normal(None)}
    if len(without_none) == 1:
        return named == next(iter(without_none))
    return named == ("union", without_none)
