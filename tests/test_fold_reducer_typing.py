"""Static typing assertions for ``FoldReducer`` generics.

These are checked by **mypy**, not pytest — the bodies are typing-only.
Run: ``uv run mypy tests/test_fold_reducer_typing.py``

A passing mypy run proves two things at once:
- the ``assert_type`` calls confirm ``S`` is inferred (positive checks);
- the ``# type: ignore[arg-type]`` on the mismatch is *used* (negative check) —
  under strict ``warn_unused_ignores`` an unused ignore is itself an error, so
  the run only succeeds if the mismatch is genuinely rejected.
"""

from __future__ import annotations

from typing import assert_type

from langgraph_events import Foldable, FoldReducer, IntegrationEvent


class Incremented(IntegrationEvent):
    by: int = 1

    def fold(self, state: dict[str, int]) -> dict[str, int]:
        return {"n": state["n"] + self.by}


def infers_state_type_from_default_factory() -> None:
    counter = FoldReducer(
        name="counter",
        event_type=Incremented,
        default_factory=lambda: {"n": 0},
    )
    assert_type(counter.empty, dict[str, int])
    assert_type(counter.seed([]), dict[str, int])


def infers_state_type_for_explicit_fold() -> None:
    # An explicit fold types its state against S; the event is annotated
    # against the structural Foldable contract.
    def step(state: dict[str, int], event: Foldable) -> dict[str, int]:
        return event.fold(state)

    counter = FoldReducer(
        name="counter",
        event_type=Incremented,
        default_factory=lambda: {"n": 0},
        fold=step,
    )
    assert_type(counter.empty, dict[str, int])


def rejects_state_shape_mismatch() -> None:
    # With S pinned to dict[str, int], a fold whose return disagrees (int) is
    # a type error. (Without an explicit S, mypy widens S to the join of the
    # default_factory and fold return types, so pin it to assert the check.)
    FoldReducer[dict[str, int]](
        name="bad",
        event_type=Incremented,
        default_factory=lambda: {"n": 0},
        fold=lambda state, event: 5,  # type: ignore[arg-type, return-value]
    )
