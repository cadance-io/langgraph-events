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


def _alias_kinds() -> tuple[type, ...]:
    """The classes of a type alias statement: ``typing.TypeAliasType`` on
    Python 3.12 and later, and ``typing_extensions.TypeAliasType``."""
    kinds = [getattr(typing, "TypeAliasType", None)]
    try:
        import typing_extensions  # noqa: PLC0415
    except ImportError:  # pragma: no cover - a dependency of langgraph
        pass
    else:
        kinds.append(typing_extensions.TypeAliasType)
    return tuple(k for k in kinds if isinstance(k, type))


_ALIAS_KINDS = _alias_kinds()


def _unaliased(hint: object) -> object:
    """*hint* with a type alias replaced by its value.

    A subscripted alias such as ``G[int]`` for ``type G[T] = list[T]``
    gives ``list[int]``. Raises :class:`UnsettledTypeError` for a bare alias
    with type parameters, and for an alias whose value cannot take its
    arguments.
    """
    if isinstance(hint, _ALIAS_KINDS):
        if getattr(hint, "__type_params__", ()):
            raise UnsettledTypeError(
                f"the type alias {getattr(hint, '__name__', hint)} has type "
                f"parameters. Subscript it, for example "
                f"{getattr(hint, '__name__', hint)}[int]"
            )
        return hint.__value__  # type: ignore[attr-defined]
    alias = typing.get_origin(hint)
    if not isinstance(alias, _ALIAS_KINDS):
        return hint
    params = tuple(getattr(alias, "__type_params__", ()))
    args = typing.get_args(hint)
    value = alias.__value__  # type: ignore[attr-defined]
    order = tuple(getattr(value, "__parameters__", ()))
    if len(params) != len(args) or set(order) != set(params):
        raise UnsettledTypeError(
            f"the type alias {getattr(alias, '__name__', alias)} cannot take the "
            f"arguments {args}. "
            f"Name its value instead"
        )
    by_param = dict(zip(params, args, strict=True))
    return value[tuple(by_param[p] for p in order)]


def _normal(hint: object) -> tuple[Any, ...]:  # noqa: PLR0911 - one return per row of the rule table
    """The canonical form of *hint*, for an exact comparison.

    Two annotations name the same type when their canonical forms are equal.
    The rule table:

    - ``Annotated`` metadata is stripped, at any depth.
    - ``None`` is ``NoneType``. ``Optional[X]`` is ``X | None``.
    - A union is the set of its members, so the order does not count. A
      member that is a union itself, behind ``Annotated`` or an alias, is
      flattened into it. The ``Literal`` members of a union merge into one
      ``Literal``, and ``None`` in a ``Literal`` is ``NoneType``.
    - A ``Literal`` is the set of its values, keyed by type and value, so
      ``Literal[1]`` is not ``Literal[True]``.
    - A generic alias is its origin class and the canonical form of each
      argument, a ``Callable`` argument list included. So ``List[int]`` is
      ``list[int]``, and ``typing.Sequence[int]`` is
      ``collections.abc.Sequence[int]``. A bare alias such as ``List`` is
      its origin class.
    - ``Any`` equals only ``Any``. A ``NewType`` is its own type.
    - A type alias statement is its value. A subscripted alias is its value
      with the arguments in place.

    Raises :class:`UnsettledTypeError` for a type variable, a forward
    reference, a string, or an alias that cannot be replaced.
    """
    hint = _unaliased(hint)
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
    if origin is Literal or origin is typing.Union or origin is types.UnionType:
        return _union_of(_members(hint))
    if origin is tuple and not args and hint is not typing.Tuple:  # noqa: UP006 - the bare alias is the case
        return ("generic", tuple, ())  # ``tuple[()]``: the empty tuple
    if origin is not None and not args:
        return ("class", origin)
    if origin is not None:
        return ("generic", origin, tuple(_normal_argument(a) for a in args))
    return ("class", hint)


def _members(hint: object) -> set[tuple[Any, ...]]:
    """The canonical members of a union or a ``Literal``, flattened.

    A ``Literal`` gives one ``literal`` member for its values, and the
    ``NoneType`` class for a ``None`` value.
    """
    origin = typing.get_origin(hint)
    if origin is Literal:
        values = typing.get_args(hint)
        members: set[tuple[Any, ...]] = {
            ("literal", frozenset((type(v), v) for v in values if v is not None))
        }
        if any(v is None for v in values):
            members.add(("class", type(None)))
        return {m for m in members if m != ("literal", frozenset())}
    found: set[tuple[Any, ...]] = set()
    for arg in typing.get_args(hint):
        normal = _normal(arg)
        found |= set(normal[1]) if normal[0] == "union" else {normal}
    return found


def _union_of(members: set[tuple[Any, ...]]) -> tuple[Any, ...]:
    """The canonical form of a union of *members*: the ``Literal`` members
    merge into one, and a union of one member is that member."""
    literals = {m for m in members if m[0] == "literal"}
    if len(literals) > 1:
        members = (members - literals) | {
            ("literal", frozenset().union(*(m[1] for m in literals)))
        }
    if len(members) == 1:
        return next(iter(members))
    return ("union", frozenset(members))


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
    return named == _union_of(set(wanted[1]) - {_normal(None)})
