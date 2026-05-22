"""IntegrationEvent must live at module level.

An ``IntegrationEvent`` crosses a context / system boundary by definition;
nesting it inside a ``Namespace`` or ``Command`` is a category error. The
rule fires in ``_NestedEventMeta.__set_name__``, so a violation surfaces as
a ``RuntimeError`` wrapping the underlying ``TypeError`` (same shape as the
Command-nested-in-Command rejection).
"""

import pytest

from langgraph_events import (
    Auditable,
    Command,
    IntegrationEvent,
    MessageEvent,
    Namespace,
)


def describe_integration_event_placement():

    def describe_rejected():

        def when_nested_in_a_Namespace():
            def it_raises_TypeError():
                with pytest.raises(RuntimeError) as exc_info:

                    class _IENsHost(Namespace):
                        class PaymentConfirmed(IntegrationEvent):
                            txn: str = ""

                cause = exc_info.value.__cause__
                assert isinstance(cause, TypeError)
                assert "module level" in str(cause)

        def when_nested_in_a_Command():
            def it_raises_TypeError():
                with pytest.raises(RuntimeError) as exc_info:

                    class _IECmdHost(Namespace):
                        class Place(Command):
                            class Paid(IntegrationEvent):
                                pass

                            def handle(self): ...

                cause = exc_info.value.__cause__
                assert isinstance(cause, TypeError)
                assert "module level" in str(cause)

        def when_nested_anywhere():
            def it_names_the_offending_class_and_owner_in_the_message():
                with pytest.raises(RuntimeError) as exc_info:

                    class _IEShippingHost(Namespace):
                        class Dispatched(IntegrationEvent):
                            pass

                cause = exc_info.value.__cause__
                assert isinstance(cause, TypeError)
                msg = str(cause)
                assert "Dispatched" in msg and "_IEShippingHost" in msg

    def describe_accepted():

        def when_declared_at_module_level():
            def it_loads():
                class PaymentConfirmed(IntegrationEvent):
                    txn: str = ""

                assert PaymentConfirmed(txn="t-1").txn == "t-1"

        def when_subclassing_a_module_level_IntegrationEvent():
            def it_loads():
                class BasePayment(IntegrationEvent):
                    txn: str = ""

                class RefundIssued(BasePayment):
                    amount: int = 0

                assert RefundIssued(txn="t-2", amount=5).amount == 5

        def when_composed_at_module_level():

            def with_behavioural_mixins():
                def it_loads():
                    class UserMessageReceived(
                        IntegrationEvent, MessageEvent, Auditable
                    ):
                        text: str = ""

                    assert UserMessageReceived(text="hi").text == "hi"
