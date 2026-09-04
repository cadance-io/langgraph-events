"""Static typing assertions for Pydantic-generated ``Event`` constructors.

Checked by **mypy**, not pytest:
``uv run mypy tests/test_event_typing.py``.

Without the transform a type checker cannot see fields inherited through
python-event-sourcery's base, so consumer constructors reject every keyword.
Strict ``warn_unused_ignores`` makes each negative assertion prove that mypy
really rejects the invalid call.
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


def the_package_reexports_its_documented_public_names() -> None:
    """``Event`` and ``HandlerReturn`` are importable by design; the docs use both.

    Under ``no_implicit_reexport`` a name the package imports but never re-exports is
    invisible to a consumer, however public its docstring claims to be.
    """

    def catch_all(event: Event) -> HandlerReturn:
        return None

    assert_type(catch_all(Ordered()), HandlerReturn)
