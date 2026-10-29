"""Stable identities of a class: the command identity and the stored identity.

A single source of truth so the command identity used for the graph-node /
checkpoint name (``_handler``), command-privacy diagnostics, and smell messages
never drift apart. Pure string transform — imports nothing from the package, so
it is safe to import from any layer without a cycle.
"""

from __future__ import annotations

from typing import NamedTuple


def command_identity(cls: type) -> str:
    """Return ``cls.__qualname__`` with the ``<locals>.`` marker stripped.

    Function-/test-local classes carry ``<locals>`` segments in their qualname
    (e.g. ``outer.<locals>.Order.Place``); strip them so the identity reads as
    the bare nesting path (``Order.Place``) wherever it is surfaced to users.
    """
    return cls.__qualname__.replace("<locals>.", "")


class Identity(NamedTuple):
    """The stored name of an event class: its module and its qualname."""

    module: str
    qualname: str

    @property
    def label(self) -> str:
        """``<module>:<qualname>``, the form every message and report uses."""
        return f"{self.module}:{self.qualname}"
