"""DDD Expense Approval Agent — langgraph-events demo.

Demonstrates DDD domains combined with human-in-the-loop approval:

- ``Namespace`` with ``ScalarReducer`` — the ``Expense`` namespace tracks
  status through the approval lifecycle.
- Inline ``handle`` with LLM — ``Submit.handle`` calls an LLM to extract
  structured expense data from a natural-language description.
- ``Interrupted`` inside a DDD flow — when the expense exceeds the policy
  threshold, the graph pauses for manager review.
- ``resume()`` with a ``Command`` as seed — the manager resumes the graph
  by sending an ``Approve`` or ``Reject`` command, which the domain
  processes through its inline handler.

Usage:
    export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
    python examples/expense_approval.py
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    Interrupted,
    Namespace,
    ScalarReducer,
    on,
)

# ---------------------------------------------------------------------------
# LLM — structured output for expense extraction
# ---------------------------------------------------------------------------

APPROVAL_THRESHOLD = 500


class ExpenseData(BaseModel):
    """LLM output schema for expense extraction."""

    amount: float = Field(description="Total expense amount in USD")
    category: str = Field(
        description="Expense category: meals, travel, supplies, entertainment, other",
    )
    merchant: str = Field(description="Name of the merchant or vendor")
    valid: bool = Field(
        description="Whether the description is a recognisable business expense",
    )
    reason: str = Field(
        default="",
        description="If invalid, the reason why",
    )


llm = ChatOpenAI(model="gpt-4o-mini")
expense_llm = llm.with_structured_output(ExpenseData)


# ---------------------------------------------------------------------------
# Namespace: Expense
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "Submitted": "submitted",
    "Invalidated": "invalid",
    "Approved": "approved",
    "Rejected": "rejected",
}


class Expense(Namespace):
    """Expense report lifecycle.

    The ``status`` reducer tracks the current state. Only ``DomainEvent``
    outcomes update it — intermediate ``Command`` dispatches are ignored.
    """

    status = ScalarReducer(
        event_type=DomainEvent,
        fn=lambda e: _STATUS_MAP.get(type(e).__name__, "unknown"),
        default="draft",
    )

    class Submit(Command):
        """Submit an expense from a natural-language description.

        The inline ``handle`` calls an LLM to extract structured fields
        (amount, category, merchant) from free text.
        """

        description: str = ""

        class Submitted(DomainEvent):
            """Expense parsed and accepted for review."""

            amount: float = 0.0
            category: str = ""
            merchant: str = ""

        class Invalidated(DomainEvent):
            """Description could not be parsed as a valid expense."""

            reason: str = ""

        def handle(self) -> Expense.Submit.Submitted | Expense.Submit.Invalidated:
            result = expense_llm.invoke(
                f"Extract the expense details from this description. "
                f"Mark it invalid if it is not a recognisable business "
                f"expense.\n\n"
                f"Description: {self.description}"
            )
            if not result.valid:
                return Expense.Submit.Invalidated(reason=result.reason)
            return Expense.Submit.Submitted(
                amount=result.amount,
                category=result.category,
                merchant=result.merchant,
            )

    class Approve(Command):
        """Manager approves the expense."""

        approver: str = ""

        class Approved(DomainEvent):
            """Expense approved for reimbursement."""

            approver: str = ""

        def handle(self) -> Expense.Approve.Approved:
            return Expense.Approve.Approved(approver=self.approver)

    class Reject(Command):
        """Manager rejects the expense."""

        reason: str = ""

        class Rejected(DomainEvent):
            """Expense rejected."""

            reason: str = ""

        def handle(self) -> Expense.Reject.Rejected:
            return Expense.Reject.Rejected(reason=self.reason)


# ---------------------------------------------------------------------------
# Interrupted event — pauses the graph for manager review
# ---------------------------------------------------------------------------


class ApprovalRequired(Interrupted):
    """Pause the graph until a manager approves or rejects."""

    amount: float = 0.0
    summary: str = ""


# ---------------------------------------------------------------------------
# External handler — policy checker
# ---------------------------------------------------------------------------


@on
def check_policy(
    event: Expense.Submit.Submitted,
) -> Expense.Approve | ApprovalRequired:
    """Enforce the company expense policy.

    Expenses at or below the threshold are auto-approved. Expenses above
    the threshold pause the graph for manager review via ``Interrupted``.
    """
    if event.amount <= APPROVAL_THRESHOLD:
        return Expense.Approve(approver="auto-policy")
    return ApprovalRequired(
        amount=event.amount,
        summary=f"{event.category} at {event.merchant}: ${event.amount:.2f}",
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = EventGraph.from_namespaces(
    Expense,
    handlers=[check_policy],
    checkpointer=MemorySaver(),
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Namespace ===")
    print(graph.namespaces().text())
    print()

    # Scenario 1 — auto-approved (under threshold)
    print(f"=== Auto-approved (under ${APPROVAL_THRESHOLD}) ===")
    log1 = graph.invoke(
        Expense.Submit(description="Coffee with client at Starbucks, $12"),
        config={"configurable": {"thread_id": "exp-001"}},
    )
    for ev in log1:
        print(f"  {type(ev).__qualname__}: {ev}")
    print()

    # Scenario 2 — manager approves (over threshold)
    print(f"=== Manager approval (over ${APPROVAL_THRESHOLD}) ===")
    config_2 = {"configurable": {"thread_id": "exp-002"}}
    log2 = graph.invoke(
        Expense.Submit(description="Team dinner at Nobu, $750"),
        config=config_2,
    )
    print("  --- graph paused for approval ---")
    for ev in log2:
        print(f"  {type(ev).__qualname__}: {ev}")

    # Manager approves — resume with an Approve command
    print("  --- manager approves ---")
    log2 = graph.resume(
        Expense.Approve(approver="manager@co.com"),
        config=config_2,
    )
    for ev in log2:
        print(f"  {type(ev).__qualname__}: {ev}")
    print()

    # Scenario 3 — manager rejects (over threshold)
    print(f"=== Manager rejection (over ${APPROVAL_THRESHOLD}) ===")
    config_3 = {"configurable": {"thread_id": "exp-003"}}
    log3 = graph.invoke(
        Expense.Submit(description="Luxury spa day, $1200"),
        config=config_3,
    )
    print("  --- graph paused for approval ---")

    # Manager rejects — resume with a Reject command
    print("  --- manager rejects ---")
    log3 = graph.resume(
        Expense.Reject(reason="Not a business expense"),
        config=config_3,
    )
    for ev in log3:
        print(f"  {type(ev).__qualname__}: {ev}")


if __name__ == "__main__":
    main()
