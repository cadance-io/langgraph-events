"""Naming two classes in one diagnostic so they can be told apart.

A message that compares classes by identity but renders them by ``__name__``
prints one string twice whenever the two happen to share a name — "declares
`Placed` but does not cover `Placed`", or "`Trading` and `Trading`". The
reader is told nothing, and the advice that follows looks already satisfied.

:func:`distinct_labels` escalates only as far as it must: bare names when
they differ, qualnames when the names collide, and object identity when even
the qualnames match — which is exactly the two-engine-lifetimes case, where
nothing textual separates them.
"""

from __future__ import annotations


def distinct_labels(here: type, there: type) -> tuple[str, str]:
    """Render *here* and *there* so a reader can tell them apart.

    Returns equal strings when both arguments are the same class — there is
    no difference to show, and inventing one would be a lie.
    """
    if here is there:
        return here.__name__, there.__name__
    if here.__name__ != there.__name__:
        return here.__name__, there.__name__
    left = f"{here.__module__}.{here.__qualname__}"
    right = f"{there.__module__}.{there.__qualname__}"
    if left != right:
        return left, right
    return f"{left} ({id(here):#x})", f"{right} ({id(there):#x})"
