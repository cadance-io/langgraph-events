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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def escalating_labels(classes: Sequence[type]) -> dict[type, str]:
    """Label every class in *classes* just precisely enough to tell them apart.

    Bare names while those are unique; qualified names once any two collide;
    qualified names plus object identity when even those match — which is the
    two-engine-lifetimes case, where nothing textual separates them.

    One rule for every diagnostic that names classes, so a message can never
    print the same string for two different things.
    """
    unique = list(dict.fromkeys(classes))
    if len({c.__name__ for c in unique}) == len(unique):
        return {c: c.__name__ for c in unique}
    qualified = {c: f"{c.__module__}.{c.__qualname__}" for c in unique}
    if len(set(qualified.values())) == len(unique):
        return qualified
    return {c: f"{label} ({id(c):#x})" for c, label in qualified.items()}


def distinct_labels(here: type, there: type) -> tuple[str, str]:
    """Render *here* and *there* so a reader can tell them apart.

    Returns equal strings when both arguments are the same class — there is
    no difference to show, and inventing one would be a lie.
    """
    if here is there:
        return here.__name__, there.__name__
    labels = escalating_labels((here, there))
    return labels[here], labels[there]
