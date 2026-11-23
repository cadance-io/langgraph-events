"""Stable identities of a class: the command identity and the stored identity.

A single source of truth so the command identity used for the graph-node /
checkpoint name (``_handler``), command-privacy diagnostics, and smell messages
never drift apart. Pure string transform — imports nothing from the package, so
it is safe to import from any layer without a cycle.
"""

import importlib
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

    @property
    def owned_by_library(self) -> bool:
        """The module is ``langgraph_events`` or one of its submodules."""
        return self.module == _LIBRARY or self.module.startswith(f"{_LIBRARY}.")


def live_value_class(module: str, name: str) -> type | None:
    """The class that *module* itself defines under *name*, or ``None``.

    A class that *module* only imports does not count, because the bytes
    that name *module* then depend on an import that nothing records.
    """
    try:
        found = getattr(importlib.import_module(module), name, None)
    except Exception:  # an import failure of any type means "not live"
        return None
    if isinstance(found, type) and (found.__module__, found.__name__) == (module, name):
        return found
    return None


class ValueIdentity(NamedTuple):
    """The stored name of a value class: its module and its ``__name__``.

    LangGraph stores ``__name__``, not ``__qualname__``, and revives with
    one ``getattr`` on the module. So a class nested in another class has no
    identity that revives.
    """

    module: str
    name: str

    @classmethod
    def of(cls, value_cls: type) -> ValueIdentity:
        """The identity of *value_cls*.

        Raises:
            ValueError: when the module does not bind *value_cls* under its
                ``__name__``.
        """
        if live_value_class(value_cls.__module__, value_cls.__name__) is value_cls:
            return cls(value_cls.__module__, value_cls.__name__)
        remedy = (
            f"Move {value_cls.__qualname__} to module scope in {value_cls.__module__}."
            if "." in value_cls.__qualname__
            else f"Bind it on module {value_cls.__module__} under the name "
            f"{value_cls.__name__!r}. Alias a parametrized generic at module "
            f"level, for example IntBox = Box[int]."
        )
        raise ValueError(
            f"Its checkpoint identity {value_cls.__module__}."
            f"{value_cls.__name__} does not import. {remedy}"
        )

    @property
    def label(self) -> str:
        """``<module>:<name>``, the form every message and report uses."""
        return f"{self.module}:{self.name}"


_LIBRARY = "langgraph_events"
