"""Static typing assertions for ``FoldReducer`` generics.

These are checked by **mypy**, not pytest — the bodies are typing-only.
Run: ``uv run mypy tests/test_fold_reducer_typing.py``

A passing mypy run proves two things at once:
- the ``assert_type`` calls confirm ``S`` is inferred (positive checks);
- the ``# type: ignore[arg-type]`` on the mismatch is *used* (negative check) —
  under strict ``warn_unused_ignores`` an unused ignore is itself an error, so
  the run only succeeds if the mismatch is genuinely rejected.
"""

from typing import assert_type
from uuid import UUID, uuid4

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


class Approved(IntegrationEvent):
    item: UUID


class Rejected(IntegrationEvent):
    item: UUID


def accepts_a_fold_typed_on_the_event_union() -> None:
    # E names the events that fold reads, so fold can take its real union
    # instead of Foldable or Any.
    def done_items(done: list[UUID], event: Approved | Rejected) -> list[UUID]:
        return [*done, event.item]

    done = FoldReducer[list[UUID], Approved | Rejected](
        name="done",
        event_type=(Approved, Rejected),
        default_factory=list,
        fold=done_items,
    )
    assert_type(done.empty, list[UUID])
    assert_type(done.seed([Approved(item=uuid4())]), list[UUID])


def rejects_a_fold_narrower_than_the_event_type() -> None:
    # A fold that reads only Approved cannot take every event of E.
    def approved_items(done: list[UUID], event: Approved) -> list[UUID]:
        return [*done, event.item]

    FoldReducer[list[UUID], Approved | Rejected](
        name="done",
        event_type=(Approved, Rejected),
        default_factory=list,
        fold=approved_items,  # type: ignore[arg-type]
    )


def _approved_items(done: list[UUID], event: Approved) -> list[UUID]:
    return [*done, event.item]


def rejects_an_event_type_wider_than_E() -> None:
    # event_type is typed against E, so it cannot name an event outside E.
    FoldReducer[list[UUID], Approved](
        name="done",
        event_type=(Approved, Rejected),  # type: ignore[arg-type]
        default_factory=list,
        fold=_approved_items,
    )


def rejects_swapped_type_arguments() -> None:
    # The state type comes first: FoldReducer[S, E], not FoldReducer[E, S].
    FoldReducer[Approved | Rejected, list[UUID]](
        name="done",
        event_type=(Approved, Rejected),  # type: ignore[arg-type]
        default_factory=list,  # type: ignore[arg-type]
        fold=_approved_items,  # type: ignore[arg-type]
    )


class _DoneItems(FoldReducer[list[UUID]]):
    """A subclass that leaves E out, so E is Foldable."""


def rejects_a_foldless_event_for_the_default_event_type() -> None:
    # E defaults to Foldable, and an event without a fold method is not
    # Foldable, so the reducer must name E. (mypy infers E from the arguments
    # of a call such as FoldReducer[S](...), so the subclass pins it here.)
    _DoneItems(
        name="done",
        event_type=(Approved, Rejected),  # type: ignore[arg-type]
        default_factory=list,
    )
