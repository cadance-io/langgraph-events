"""Shared validators for declaration-site arguments.

Imports nothing from the package, so it is safe to import from any layer
without a cycle — same contract as :mod:`langgraph_events._identity`.
"""

from __future__ import annotations


def normalize_exception_tuple(
    value: type[Exception] | tuple[type[Exception], ...], *, owner: str
) -> tuple[type[Exception], ...]:
    """Coerce a scalar-or-tuple of exception classes into a validated tuple.

    *owner* prefixes the error in the declaration site's voice —
    ``"@on() raises="`` for the decorator, ``"RetryPolicy on="`` for the policy
    field — so one taxonomy lives in one place while each site still names
    itself. Mirrors :func:`langgraph_events._handler.normalize_previous_names`.
    """
    normalized = value if isinstance(value, tuple) else (value,)
    for entry in normalized:
        if not (isinstance(entry, type) and issubclass(entry, Exception)):
            raise TypeError(
                f"{owner} entries must be Exception subclasses, got {entry!r}. "
                f"Non-Exception BaseException subclasses (KeyboardInterrupt, "
                f"SystemExit, GeneratorExit, asyncio.CancelledError) are not "
                f"allowed — they are runtime/exit signals, not domain errors."
            )
    return normalized
