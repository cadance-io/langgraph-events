"""Static typing assertions for the ``@dataclass_transform`` on ``Event``.

These are checked by **mypy**, not pytest — the bodies are typing-only.
Run: ``uv run mypy tests/test_event_typing.py``

``Event.__init_subclass__`` turns every subclass into a frozen dataclass at runtime.
Without the transform a type checker cannot see that, so a consumer's constructor is
invisible and every construction has to be silenced. A passing mypy run proves three
things, three of them negative — under strict ``warn_unused_ignores`` an unused
ignore is itself an error, so an ignore only survives if what it silences is
genuinely rejected.
"""

from __future__ import annotations

from typing import assert_type

from langgraph_events import Event, HandlerReturn, IntegrationEvent


class Ordered(IntegrationEvent):
    sku: str = ""
    quantity: int = 1


def constructor_is_visible_and_fields_keep_their_types() -> None:
    event = Ordered(sku="abc", quantity=2)
    assert_type(event.sku, str)
    assert_type(event.quantity, int)


def constructor_rejects_an_unknown_field() -> None:
    Ordered(nope=1)  # type: ignore[call-arg]


def constructor_rejects_a_mistyped_field() -> None:
    Ordered(quantity="two")  # type: ignore[arg-type]


def events_are_frozen() -> None:
    event = Ordered(sku="abc")
    event.sku = "def"  # type: ignore[misc]


def the_package_reexports_its_documented_public_names() -> None:
    """``Event`` and ``HandlerReturn`` are importable by design; the docs use both.

    Under ``no_implicit_reexport`` a name the package imports but never re-exports is
    invisible to a consumer, however public its docstring claims to be.
    """

    def catch_all(event: Event) -> HandlerReturn:
        return None

    assert_type(catch_all(Ordered()), HandlerReturn)
