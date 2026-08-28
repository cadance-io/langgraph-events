"""Warning emission anchored at user code.

``warnings.warn`` takes a ``stacklevel`` counted in frames from the call
site. Hand-counting it is fragile in a library: the number is correct only
for the exact call depth it was written at, and silently goes stale the
moment the emitting code moves into a helper or gains a caller. Nothing
fails — the warning just starts pointing at library internals, or at the
importing module's ``import`` line, and the user is none the wiser.

:func:`warn_user` computes the depth instead: it walks out to the first
frame that does not belong to this package and anchors there.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent
"""Frames under this directory are library frames, and never the anchor."""


def warn_user(
    message: str,
    category: type[Warning] = UserWarning,
    *,
    extra_depth: int = 0,
) -> None:
    """Emit *message* anchored at the nearest frame outside this package.

    ``extra_depth`` skips that many additional frames, for the rare case
    where the immediate caller is itself user code that should not be
    blamed — a class body executing during ``__init_subclass__``, say.
    """
    depth, frame = 2, inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    while frame is not None and _is_library_frame(frame.f_code.co_filename):
        frame, depth = frame.f_back, depth + 1
    warnings.warn(message, category, stacklevel=depth + extra_depth)


def _is_library_frame(filename: str) -> bool:
    """True when *filename* lives inside this package."""
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_ROOT.resolve())
    except (OSError, ValueError):
        # Synthetic frames (``<stdin>``, ``<string>``) are never library code.
        return False
