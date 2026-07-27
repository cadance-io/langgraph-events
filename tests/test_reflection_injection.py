"""Tests for injecting Reflection into handlers by parameter annotation."""

from __future__ import annotations

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Namespace,
    Reflection,
    on,
)

_CAPTURED: list[Reflection] = []
_LOG_SIZES: list[tuple[int, int]] = []


class Billing(Namespace):
    class Charge(Command):
        amount: int = 0

        class Charged(DomainEvent):
            amount: int = 0

        def charge(self) -> Billing.Charge.Charged:
            return Billing.Charge.Charged(amount=self.amount)


@on(Billing.Charge.Charged)
def audit(event: Billing.Charge.Charged, run: Reflection) -> None:
    _CAPTURED.append(run)


@on(Billing.Charge.Charged)
def compare(event: Billing.Charge.Charged, log: EventLog, run: Reflection) -> None:
    _LOG_SIZES.append((len(log), len(run.log)))


def describe_reflection_injection():
    def it_injects_a_reflection_snapshot_mid_run():
        _CAPTURED.clear()
        graph = EventGraph([Billing.Charge, audit])

        graph.invoke(Billing.Charge(amount=7))

        assert len(_CAPTURED) == 1
        snapshot = _CAPTURED[0]
        assert isinstance(snapshot, Reflection)
        assert "Charged" in snapshot.tool().run(op="filter", type="Charged")

    def it_coexists_in_one_handler_alongside_event_log_injection():
        _LOG_SIZES.clear()
        graph = EventGraph([Billing.Charge, compare])

        graph.invoke(Billing.Charge(amount=7))

        assert _LOG_SIZES == [(2, 2)]

    def it_reuses_a_cached_namespace_model_across_calls():
        graph = EventGraph([Billing.Charge, audit])

        assert graph.namespaces() is graph.namespaces()
