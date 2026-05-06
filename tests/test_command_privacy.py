"""Tests for the Command-privacy invariant enforced at ``EventGraph`` build.

Two symmetric rules:

- A ``Command.handle()`` may only emit ``DomainEvent``s nested under that same
  Command (or a parent Command, for inheritance).
- An ``@on(...)`` reactor may only emit namespace-level ``DomainEvent``s —
  Command-private outcomes are off-limits.
"""

from __future__ import annotations

import pytest

from langgraph_events import (
    Command,
    CommandPrivacyError,
    DomainEvent,
    EventGraph,
    Interrupted,
    Namespace,
    Scatter,
    on,
)

# ---- Module-level fixtures -------------------------------------------------
# Annotations on Command.handle() and @on handlers in the test bodies below
# resolve via the function's __globals__, which is *this* module — so any
# class referenced from those annotations must live at module level.


class _PrivSib(Namespace):
    """Command emits a sibling event (illegal)."""

    class DoIt(Command):
        class Done(DomainEvent):
            pass

        def handle(self) -> _PrivSib.DoIt.Done | _PrivSib.Stray:
            return _PrivSib.DoIt.Done()

    class Stray(DomainEvent):
        pass


class _PrivForeign(Namespace):
    """One Command emits another Command's private outcome (illegal)."""

    class A(Command):
        class ADone(DomainEvent):
            pass

        def handle(self) -> _PrivForeign.A.ADone | _PrivForeign.B.BDone:
            return _PrivForeign.A.ADone()

    class B(Command):
        class BDone(DomainEvent):
            pass

        def handle(self) -> _PrivForeign.B.BDone:
            return _PrivForeign.B.BDone()


class _PrivScat(Namespace):
    """Command's handle returns Scatter of a sibling (illegal)."""

    class Burst(Command):
        class Pulse(DomainEvent):
            pass

        def handle(self) -> Scatter[_PrivScat.Loose] | _PrivScat.Burst.Pulse:
            return Scatter([_PrivScat.Loose()])

    class Loose(DomainEvent):
        pass


class _PrivClean(Namespace):
    """All-positive: Command emits only its own nested outcomes."""

    class Tick(Command):
        class Ticked(DomainEvent):
            pass

        def handle(self) -> _PrivClean.Tick.Ticked:
            return _PrivClean.Tick.Ticked()


class _PrivBare(Namespace):
    """Command with no return annotation — exempt."""

    class DoIt(Command):
        class Done(DomainEvent):
            pass

        def handle(self):
            return _PrivBare.DoIt.Done()


class _PrivPause(Namespace):
    """Command returns Interrupted — framework event, exempt."""

    class Wait(Command):
        class Resumed(DomainEvent):
            pass

        def handle(self) -> _PrivPause.Wait.Resumed | Interrupted:
            return Interrupted(reason="waiting")


class _PrivInherit(Namespace):
    """Child Command inherits its parent's private outcome (legal)."""

    class Parent(Command):
        class Done(DomainEvent):
            pass

        def handle(self) -> _PrivInherit.Parent.Done:
            return _PrivInherit.Parent.Done()

    class Child(Parent):
        def handle(self) -> _PrivInherit.Parent.Done:
            return _PrivInherit.Parent.Done()


class _PrivLeaky(Namespace):
    """Reactor that emits a Command-private event from outside its owner."""

    class Persist(Command):
        class Persisted(DomainEvent):
            pass

        def handle(self) -> _PrivLeaky.Persist.Persisted:
            return _PrivLeaky.Persist.Persisted()

    class Trigger(DomainEvent):
        pass


class _PrivShared(Namespace):
    """Reactor emits a sibling DomainEvent (legal)."""

    class Persist(Command):
        class Persisted(DomainEvent):
            pass

        def handle(self) -> _PrivShared.Persist.Persisted:
            return _PrivShared.Persist.Persisted()

    class Note(DomainEvent):
        pass

    class Reminder(DomainEvent):
        pass


def describe_command_privacy():
    def describe_command_handle():
        def when_handle_returns_a_namespace_level_sibling_event():
            def it_raises_CommandPrivacyError():
                with pytest.raises(CommandPrivacyError, match=r"_PrivSib\.Stray"):
                    EventGraph([_PrivSib.DoIt])

        def when_handle_returns_an_event_owned_by_a_different_command():
            def it_raises_CommandPrivacyError():
                with pytest.raises(
                    CommandPrivacyError, match=r"private to _PrivForeign\.B"
                ):
                    EventGraph([_PrivForeign.A, _PrivForeign.B])

        def when_handle_returns_Scatter_of_a_non_nested_event():
            def it_raises_CommandPrivacyError():
                with pytest.raises(CommandPrivacyError, match=r"_PrivScat\.Loose"):
                    EventGraph([_PrivScat.Burst])

        def when_handle_returns_only_nested_outcomes():
            def it_builds_the_graph():
                EventGraph([_PrivClean.Tick])  # no raise

        def when_handle_has_no_return_annotation():
            def it_builds_the_graph():
                EventGraph([_PrivBare.DoIt])  # no raise

        def when_handle_returns_Interrupted():
            def it_builds_the_graph():
                EventGraph([_PrivPause.Wait])  # no raise

        def when_child_command_handle_returns_a_private_event_of_its_parent():
            def it_builds_the_graph():
                EventGraph([_PrivInherit.Child])  # no raise

    def describe_reactor():
        def when_reactor_returns_a_command_private_event():
            def it_raises_CommandPrivacyError():
                @on(_PrivLeaky.Trigger)
                def leak(event: _PrivLeaky.Trigger) -> _PrivLeaky.Persist.Persisted:
                    return _PrivLeaky.Persist.Persisted()

                with pytest.raises(
                    CommandPrivacyError, match=r"private to _PrivLeaky\.Persist"
                ):
                    EventGraph([_PrivLeaky.Persist, leak])

        def when_reactor_returns_Scatter_of_a_command_private_event():
            def it_raises_CommandPrivacyError():
                @on(_PrivLeaky.Trigger)
                def burst(
                    event: _PrivLeaky.Trigger,
                ) -> Scatter[_PrivLeaky.Persist.Persisted]:
                    return Scatter([_PrivLeaky.Persist.Persisted()])

                with pytest.raises(
                    CommandPrivacyError, match=r"private to _PrivLeaky\.Persist"
                ):
                    EventGraph([_PrivLeaky.Persist, burst])

        def when_reactor_scatters_a_Command_private_event_via_bare_Scatter():
            # Bare ``-> Scatter`` is a legitimate annotation when the
            # handler scatters non-private events, so we don't reject it at
            # build time. Privacy is enforced at runtime instead, when the
            # reactor actually emits a Command-private event.
            def it_raises_CommandPrivacyError_at_dispatch():
                @on(_PrivLeaky.Trigger)
                def burst(event: _PrivLeaky.Trigger) -> Scatter:
                    return Scatter([_PrivLeaky.Persist.Persisted()])

                graph = EventGraph([_PrivLeaky.Persist, burst])
                with pytest.raises(
                    CommandPrivacyError,
                    match=r"private to _PrivLeaky\.Persist",
                ):
                    graph.invoke(_PrivLeaky.Trigger())

        def when_reactor_returns_a_namespace_level_event():
            def it_builds_the_graph():
                @on(_PrivShared.Note)
                def echo(event: _PrivShared.Note) -> _PrivShared.Reminder:
                    return _PrivShared.Reminder()

                EventGraph([_PrivShared.Persist, echo])  # no raise

    def describe_error():
        def it_subclasses_TypeError():
            assert issubclass(CommandPrivacyError, TypeError)

        def it_names_the_handler_and_the_event_in_the_message():
            with pytest.raises(CommandPrivacyError) as exc_info:
                EventGraph([_PrivSib.DoIt])
            msg = str(exc_info.value)
            assert "_PrivSib.DoIt" in msg and "_PrivSib.Stray" in msg
