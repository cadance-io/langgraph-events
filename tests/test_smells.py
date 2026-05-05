"""Tests for the domain-pattern smell warning."""

from __future__ import annotations

import warnings

from langgraph_events import (
    Command,
    DomainEvent,
    DomainPatternWarning,
    EventGraph,
    Namespace,
    on,
)


# Two sources in the same namespace fan out to identical target set via two
# distinct handlers — exact match for the smell detection rule.
class _SmellPair(Namespace):
    class TriggerA(Command):
        class Done(DomainEvent):
            pass

    class TriggerB(Command):
        class Done(DomainEvent):
            pass

    class Out1(DomainEvent):
        pass

    class Out2(DomainEvent):
        pass


@on(_SmellPair.TriggerA)
def smell_pair_a(event: _SmellPair.TriggerA) -> _SmellPair.Out1 | _SmellPair.Out2:
    return _SmellPair.Out1()


@on(_SmellPair.TriggerB)
def smell_pair_b(event: _SmellPair.TriggerB) -> _SmellPair.Out1 | _SmellPair.Out2:
    return _SmellPair.Out1()


_SMELL_PAIR_HANDLERS = [smell_pair_a, smell_pair_b]


# A solitary fanout — one source, fan-out to many targets — must NOT trigger.
class _SmellSolo(Namespace):
    class Trigger(Command):
        class Done(DomainEvent):
            pass

    class Out1(DomainEvent):
        pass

    class Out2(DomainEvent):
        pass

    class Out3(DomainEvent):
        pass


@on(_SmellSolo.Trigger)
def smell_solo_handler(
    event: _SmellSolo.Trigger,
) -> _SmellSolo.Out1 | _SmellSolo.Out2 | _SmellSolo.Out3:
    return _SmellSolo.Out1()


_SMELL_SOLO_HANDLERS = [smell_solo_handler]


# Overlapping but unequal target sets — must NOT trigger (subset/superset rule).
class _SmellOverlap(Namespace):
    class TriggerA(Command):
        class Done(DomainEvent):
            pass

    class TriggerB(Command):
        class Done(DomainEvent):
            pass

    class Shared(DomainEvent):
        pass

    class OnlyA(DomainEvent):
        pass

    class OnlyB(DomainEvent):
        pass


@on(_SmellOverlap.TriggerA)
def smell_overlap_a(
    event: _SmellOverlap.TriggerA,
) -> _SmellOverlap.Shared | _SmellOverlap.OnlyA:
    return _SmellOverlap.Shared()


@on(_SmellOverlap.TriggerB)
def smell_overlap_b(
    event: _SmellOverlap.TriggerB,
) -> _SmellOverlap.Shared | _SmellOverlap.OnlyB:
    return _SmellOverlap.Shared()


_SMELL_OVERLAP_HANDLERS = [smell_overlap_a, smell_overlap_b]


# Two distinct patterns in two different namespaces — should warn twice.
class _SmellTwoNs1(Namespace):
    class T1A(Command):
        class Done(DomainEvent):
            pass

    class T1B(Command):
        class Done(DomainEvent):
            pass

    class P(DomainEvent):
        pass

    class Q(DomainEvent):
        pass


class _SmellTwoNs2(Namespace):
    class T2A(Command):
        class Done(DomainEvent):
            pass

    class T2B(Command):
        class Done(DomainEvent):
            pass

    class R(DomainEvent):
        pass

    class S(DomainEvent):
        pass


@on(_SmellTwoNs1.T1A)
def smell_two_ns1_a(event: _SmellTwoNs1.T1A) -> _SmellTwoNs1.P | _SmellTwoNs1.Q:
    return _SmellTwoNs1.P()


@on(_SmellTwoNs1.T1B)
def smell_two_ns1_b(event: _SmellTwoNs1.T1B) -> _SmellTwoNs1.P | _SmellTwoNs1.Q:
    return _SmellTwoNs1.P()


@on(_SmellTwoNs2.T2A)
def smell_two_ns2_a(event: _SmellTwoNs2.T2A) -> _SmellTwoNs2.R | _SmellTwoNs2.S:
    return _SmellTwoNs2.R()


@on(_SmellTwoNs2.T2B)
def smell_two_ns2_b(event: _SmellTwoNs2.T2B) -> _SmellTwoNs2.R | _SmellTwoNs2.S:
    return _SmellTwoNs2.R()


_SMELL_TWO_PATTERNS_HANDLERS = [
    smell_two_ns1_a,
    smell_two_ns1_b,
    smell_two_ns2_a,
    smell_two_ns2_b,
]


def _capture_smell_warnings(handlers: list) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        EventGraph(handlers).namespaces()
        return [w for w in captured if issubclass(w.category, DomainPatternWarning)]


def describe_domain_pattern_warning():
    def when_two_sources_share_target_set():
        def it_emits_one_warning():
            captured = _capture_smell_warnings(_SMELL_PAIR_HANDLERS)
            assert len(captured) == 1

        def it_includes_namespace_source_handler_names_and_targets_in_message():
            captured = _capture_smell_warnings(_SMELL_PAIR_HANDLERS)
            assert len(captured) == 1
            msg = str(captured[0].message)
            # Namespace name appears
            assert "_SmellPair" in msg
            # Both source class names appear
            assert "TriggerA" in msg
            assert "TriggerB" in msg
            # Both handler names appear
            assert "smell_pair_a" in msg
            assert "smell_pair_b" in msg
            # Both target class names appear
            assert "Out1" in msg
            assert "Out2" in msg

    def when_only_one_source_fans_out():
        def it_does_not_warn():
            captured = _capture_smell_warnings(_SMELL_SOLO_HANDLERS)
            assert captured == []

    def when_target_sets_overlap_but_are_not_equal():
        def it_does_not_warn():
            captured = _capture_smell_warnings(_SMELL_OVERLAP_HANDLERS)
            assert captured == []

    def when_warning_is_silenced_via_filter():
        def it_emits_no_warning():
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                warnings.filterwarnings("ignore", category=DomainPatternWarning)
                EventGraph(_SMELL_PAIR_HANDLERS).namespaces()
            smell = [
                w for w in captured if issubclass(w.category, DomainPatternWarning)
            ]
            assert smell == []

    def when_renderers_are_called_multiple_times():
        def it_warns_once_per_model_instance():
            # Building once, then calling renderers multiple times — only one
            # warning fires. The warning anchors to model construction, so
            # .mermaid() / .text() etc. don't re-emit.
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model = EventGraph(_SMELL_PAIR_HANDLERS).namespaces()
                model.mermaid()
                model.mermaid()
                model.text()
            smell = [
                w for w in captured if issubclass(w.category, DomainPatternWarning)
            ]
            assert len(smell) == 1

    def when_two_distinct_patterns_exist():
        def it_emits_one_warning_per_pattern():
            captured = _capture_smell_warnings(_SMELL_TWO_PATTERNS_HANDLERS)
            assert len(captured) == 2
            messages = [str(w.message) for w in captured]
            joined = "\n".join(messages)
            assert "_SmellTwoNs1" in joined
            assert "_SmellTwoNs2" in joined

    def when_warning_anchors_call_site():
        def it_points_at_user_code_not_library_internals():
            # Regression guard: stacklevel on warnings.warn must walk past
            # the library frames so the warning surfaces the user's
            # `EventGraph(...).namespaces()` line, not an internal file.
            captured = _capture_smell_warnings(_SMELL_PAIR_HANDLERS)
            assert len(captured) == 1
            filename = captured[0].filename
            assert "langgraph_events" not in filename, (
                f"warning anchored to library file {filename!r}; expected "
                "user code. Check stacklevel in emit_domain_pattern_warnings."
            )
