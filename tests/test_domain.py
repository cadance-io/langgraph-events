"""Tests for EventGraph.domain() — the code-derived DomainModel and its renderers."""

from __future__ import annotations

import json

from conftest import Order

from langgraph_events import (
    Command,
    Domain,
    DomainEvent,
    EventGraph,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Invariant,
    InvariantViolated,
    Resumed,
    Scatter,
    on,
)


class CustomerExists(Invariant):
    pass


class Customer(Domain):
    class Register(Command):
        email: str = ""

        class Registered(DomainEvent):
            customer_id: str = ""


class PaymentConfirmed(IntegrationEvent):
    transaction_id: str = ""


# Module-level events/domains used by test cases whose handlers need
# `get_type_hints()` to resolve class names — classes defined inside
# describe_/when_/it_ blocks are unreachable at hint-resolution time.
class _Batch(IntegrationEvent):
    pass


class _Item(IntegrationEvent):
    pass


class _AggIR(Domain):
    class Ask(Command):
        class Requested(Interrupted):
            pass

        def handle(self) -> _AggIR.Ask.Requested:
            return _AggIR.Ask.Requested()


# Shared trivial handlers — reused across tests that just need a built graph.
# Kept at module level so test bodies stay focused on assertions; each
# `describe_` re-builds the graph from these instead of redefining them.


@on(Order.Place)
def place(event: Order.Place) -> Order.Place.Placed:
    return Order.Place.Placed()


@on(Order.Shipped)
def notify(event: Order.Shipped) -> None:
    return None


# Shared handlers for describe_invariants — hoisted so each test doesn't
# redefine the same `@on(..., invariants={CustomerExists: ...})` body.
@on(Order.Place, invariants={CustomerExists: lambda log: True})
def inv_place(event: Order.Place) -> Order.Place.Placed:
    return Order.Place.Placed()


@on(Customer.Register, invariants={CustomerExists: lambda log: True})
def inv_register(event: Customer.Register) -> Customer.Register.Registered:
    return Customer.Register.Registered()


@on(InvariantViolated, invariant=CustomerExists)
def inv_explain(event: InvariantViolated) -> None:
    return None


@on(InvariantViolated)
def inv_catch_all(event: InvariantViolated) -> None:
    return None


# A policy that produces Order.Place.Rejected indirectly, so the (Place,
# Rejected) pair has no direct flow edge and ownership-gap fill kicks in.
@on(HandlerRaised)
def explain_rejection_fixture(event: HandlerRaised) -> Order.Place.Rejected:
    return Order.Place.Rejected(reason="test")


# Module-level domains with nested Halted subtypes — keeps forward-reference
# resolution happy for handler return-type introspection.
class _Content(Domain):
    class Classify(Command):
        class Classified(DomainEvent):
            label: str = ""

    class Blocked(Halted):
        reason: str = ""


class _Review(Domain):
    class Open(Command):
        class Opened(DomainEvent):
            pass

    class Abandoned(Halted):
        reason: str = ""


def describe_domain_model_shape():
    def when_graph_has_domain():
        def with_command_handler():
            def it_groups_under_domain_then_command_then_outcomes():

                d = EventGraph([place]).domain()

                assert "Order" in d.domains
                order = d.domains["Order"]
                assert "Place" in order.commands
                outcomes = order.commands["Place"].outcomes
                assert Order.Place.Placed in outcomes
                assert Order.Place.Rejected in outcomes

        def with_single_outcome_command():
            def it_lists_the_outcome_once_not_twice():
                @on(Order.Place)
                def run(event: Order.Place) -> Order.Place.Placed:
                    return Order.Place.Placed(order_id="o1")

                d = EventGraph([run]).domain()
                outcomes = d.domains["Order"].commands["Place"].outcomes
                # Place has two outcomes; each appears exactly once.
                assert list(outcomes).count(Order.Place.Placed) == 1
                assert list(outcomes).count(Order.Place.Rejected) == 1

        def with_event_unrelated_to_any_command():
            def it_lists_under_domain_events():
                @on(Order.Shipped)
                def react(event: Order.Shipped) -> None:
                    return None

                d = EventGraph([react]).domain()
                assert Order.Shipped in d.domains["Order"].events

    def when_graph_has_integration_event():
        def it_lists_under_integration_events():
            @on(PaymentConfirmed)
            def react(event: PaymentConfirmed) -> None:
                return None

            d = EventGraph([react]).domain()
            assert PaymentConfirmed in d.integration_events

    def when_graph_has_system_event_subclass():
        def it_lists_under_system_events():
            @on(Halted)
            def react(event: Halted) -> None:
                return None

            d = EventGraph([react]).domain()
            assert Halted in d.system_events

    def when_halted_subtype_is_nested_in_domain():
        def it_lists_under_domain_events_not_system_events():
            @on(_Content.Classify)
            def classify(
                event: _Content.Classify,
            ) -> _Content.Classify.Classified | _Content.Blocked:
                return _Content.Blocked(reason="test")

            d = EventGraph([classify]).domain()
            assert _Content.Blocked in d.domains["_Content"].events
            assert _Content.Blocked not in d.system_events

        def with_mermaid_render():
            def it_uses_the_halt_class():
                @on(_Review.Open)
                def react(
                    event: _Review.Open,
                ) -> _Review.Open.Opened | _Review.Abandoned:
                    return _Review.Abandoned(reason="test")

                output = EventGraph([react]).domain().mermaid()
                assert "Abandoned([Abandoned]):::halt" in output
                assert 'subgraph _Review["_Review domain"]' in output

    def when_graph_has_no_domain_events():
        def it_returns_empty_domains_dict():
            @on(PaymentConfirmed)
            def react(event: PaymentConfirmed) -> None:
                return None

            d = EventGraph([react]).domain()
            assert d.domains == {}

    def when_graph_has_multiple_domains():
        def it_lists_each_independently():
            @on(Customer.Register)
            def register(event: Customer.Register) -> Customer.Register.Registered:
                return Customer.Register.Registered(customer_id="c1")

            d = EventGraph([place, register]).domain()
            assert set(d.domains.keys()) == {"Order", "Customer"}


def describe_reactions_classification():
    def when_handler_subscribes_to_command():
        def it_classifies_as_command_handler():

            d = EventGraph([place]).domain()
            names = [ch.name for ch in d.command_handlers]
            assert "place" in names
            assert all(ch.name != "place" for ch in d.policies)

    def when_handler_subscribes_to_domain_event():
        def it_classifies_as_policy():

            d = EventGraph([notify]).domain()
            assert any(p.name == "notify" for p in d.policies)
            assert not any(ch.name == "notify" for ch in d.command_handlers)

    def when_handler_is_inline_command_handle():
        def it_flags_inline_true():
            class AggInline(Domain):
                class Ping(Command):
                    class Pinged(DomainEvent):
                        pass

                    def handle(self) -> AggInline.Ping.Pinged:
                        return AggInline.Ping.Pinged()

            d = EventGraph([AggInline.Ping]).domain()
            inline = [ch for ch in d.command_handlers if ch.inline]
            assert len(inline) == 1
            assert inline[0].commands == (AggInline.Ping,)

    def when_handler_subscribes_to_multiple_commands():
        def it_lists_all_commands_on_handler():
            class AggMulti(Domain):
                class A(Command):
                    class Done(DomainEvent):
                        pass

                class B(Command):
                    class Done(DomainEvent):
                        pass

            @on(AggMulti.A, AggMulti.B)
            def both(event) -> AggMulti.A.Done | AggMulti.B.Done:
                return AggMulti.A.Done()

            d = EventGraph([both]).domain()
            chs = [ch for ch in d.command_handlers if ch.name == "both"]
            assert len(chs) == 1
            assert set(chs[0].commands) == {AggMulti.A, AggMulti.B}

    def when_reactions_domains_both_kinds():
        def it_preserves_registration_order():

            d = EventGraph([place, notify]).domain()
            reaction_names = [r.name for r in d.reactions]
            assert "place" in reaction_names
            assert "notify" in reaction_names


def describe_command_handler_back_reference():
    def it_lists_handler_names_on_each_command():

        d = EventGraph([place]).domain()
        assert d.domains["Order"].commands["Place"].handlers == ("place",)


def describe_edges():
    def when_handler_returns_an_event():
        def it_emits_a_solid_edge():

            d = EventGraph([place]).domain()
            solid = [e for e in d.edges if e.kind == "solid"]
            assert any(
                e.source is Order.Place
                and e.target is Order.Place.Placed
                and e.via == "place"
                for e in solid
            )

    def when_handler_declares_raises():
        def it_emits_a_raises_edge_to_handler_raised():
            class _DemoError(Exception):
                pass

            @on(Order.Place, raises=_DemoError)
            def place(event: Order.Place) -> Order.Place.Placed:
                raise _DemoError

            @on(HandlerRaised, exception=_DemoError)
            def recover(event: HandlerRaised) -> None:
                return None

            d = EventGraph([place, recover]).domain()
            raises = [e for e in d.edges if e.kind == "raises"]
            assert any(
                e.source is Order.Place
                and e.target is HandlerRaised
                and e.via == "place"
                for e in raises
            )

    def when_one_handler_produces_interrupted_and_another_subscribes_resumed():
        def it_emits_a_framework_edge_from_interrupted_to_resumed():
            @on(Resumed)
            def react(event: Resumed) -> None:
                return None

            d = EventGraph([_AggIR.Ask, react]).domain()
            framework = [e for e in d.edges if e.kind == "framework"]
            assert len(framework) == 1
            assert framework[0].source is Interrupted
            assert framework[0].target is Resumed

    def when_handler_returns_typed_scatter():
        def it_emits_a_scatter_edge():
            @on(_Batch)
            def split(event: _Batch) -> Scatter[_Item]:
                return Scatter([_Item()])

            @on(_Item)
            def handle_item(event: _Item) -> None:
                return None

            d = EventGraph([split, handle_item]).domain()
            scatter_edges = [e for e in d.edges if e.kind == "scatter"]
            assert any(e.via == "split" and e.target is _Item for e in scatter_edges)


def describe_seeds():
    def when_event_has_no_incoming_edges():
        def it_appears_in_seeds():

            d = EventGraph([place]).domain()
            assert Order.Place in d.seeds

    def when_event_is_an_edge_target():
        def it_does_not_appear_in_seeds():

            d = EventGraph([place]).domain()
            assert Order.Place.Placed not in d.seeds


def describe_reaction_flags():
    def when_handler_returns_none_annotated():
        def it_flags_side_effect_true():

            d = EventGraph([notify]).domain()
            policy = next(p for p in d.policies if p.name == "notify")
            assert policy.side_effect is True
            assert policy.has_annotation is True
            assert policy.has_untyped_scatter is False

    def when_handler_has_no_return_annotation():
        def it_flags_has_annotation_false():
            @on(Order.Place)
            def mystery(event):
                return Order.Place.Placed()

            d = EventGraph([mystery]).domain()
            ch = next(c for c in d.command_handlers if c.name == "mystery")
            assert ch.has_annotation is False

    def when_handler_returns_bare_scatter():
        def it_flags_has_untyped_scatter_true():
            @on(_Batch)
            def split(event: _Batch) -> Scatter:
                return Scatter([_Batch()])

            d = EventGraph([split]).domain()
            policy = next(p for p in d.policies if p.name == "split")
            assert policy.has_untyped_scatter is True

    def when_handler_declares_invariants():
        def it_preserves_invariant_classes():
            @on(Order.Place, invariants={CustomerExists: lambda log: True})
            def place(event: Order.Place) -> Order.Place.Placed:
                return Order.Place.Placed()

            d = EventGraph([place]).domain()
            ch = next(c for c in d.command_handlers if c.name == "place")
            assert CustomerExists in ch.invariants


def describe_invariants():
    def when_single_handler_declares_one_invariant():
        def it_surfaces_the_class_in_invariants():
            d = EventGraph([inv_place]).domain()
            assert CustomerExists in [inv.cls for inv in d.invariants]

        def it_includes_the_owning_command():
            d = EventGraph([inv_place]).domain()
            inv = next(i for i in d.invariants if i.cls is CustomerExists)
            assert Order.Place in inv.commands

        def it_includes_the_declaring_handler_name():
            d = EventGraph([inv_place]).domain()
            inv = next(i for i in d.invariants if i.cls is CustomerExists)
            assert "inv_place" in inv.declared_by

    def when_pinned_reactor_exists():
        def it_lists_the_reactor_name():
            d = EventGraph([inv_place, inv_explain]).domain()
            inv = next(i for i in d.invariants if i.cls is CustomerExists)
            assert "inv_explain" in inv.reactors

    def when_catch_all_reactor_exists():
        def it_is_not_added_to_the_reactors_list():
            d = EventGraph([inv_place, inv_catch_all]).domain()
            inv = next(i for i in d.invariants if i.cls is CustomerExists)
            assert "inv_catch_all" not in inv.reactors

    def when_multiple_handlers_declare_same_invariant():
        def it_domains_into_one_entry():
            d = EventGraph([inv_place, inv_register]).domain()
            matches = [i for i in d.invariants if i.cls is CustomerExists]
            assert len(matches) == 1
            inv = matches[0]
            assert set(inv.commands) == {Order.Place, Customer.Register}
            assert set(inv.declared_by) == {"inv_place", "inv_register"}


def _full_taxonomy_graph() -> EventGraph:
    """A graph exercising every taxonomy bucket: domain commands + events,
    integration events, system events. Used by renderer/encoder tests."""

    @on(PaymentConfirmed)
    def on_payment(event: PaymentConfirmed) -> None:
        return None

    @on(Halted)
    def on_halt(event: Halted) -> None:
        return None

    return EventGraph([place, notify, on_payment, on_halt])


def describe_text_renderer():
    def when_view_structure():
        def it_renders_domain_command_outcome_tree():

            text = EventGraph([place]).domain().text(view="structure")

            assert "Order" in text
            assert "Place" in text
            assert "Placed" in text
            assert "Rejected" in text

        def it_omits_handlers_and_policies():

            text = EventGraph([place]).domain().text(view="structure")
            assert "handlers" not in text
            assert "Policies" not in text

        def it_lists_domain_events_integration_events_and_system_events():
            text = _full_taxonomy_graph().domain().text(view="structure")
            assert "Event: Shipped" in text
            assert "Integration events:" in text
            assert "PaymentConfirmed" in text
            assert "System events:" in text
            assert "Halted" in text

    def when_view_choreography():
        def it_shows_handler_name_under_command():

            text = EventGraph([place]).domain().text()
            assert "(handlers: place)" in text

        def it_lists_policies_separately():

            text = EventGraph([notify]).domain().text()
            assert "Policies:" in text
            assert "notify" in text

        def it_lists_domain_events_integration_events_and_system_events():
            text = _full_taxonomy_graph().domain().text()
            assert "Event: Shipped" in text
            assert "Integration events:" in text
            assert "PaymentConfirmed" in text
            assert "System events:" in text
            assert "Halted" in text

    def when_graph_has_invariant():
        def it_renders_the_invariants_section():
            text = EventGraph([inv_place]).domain().text()
            assert "Invariants:" in text
            assert "CustomerExists  (on Place" in text

    def when_view_unknown():
        def it_raises_value_error():
            d = EventGraph(
                [
                    lambda event: None  # silly handler just to build a graph
                    for _ in ()
                ]
                or [_placeholder_handler()]
            ).domain()
            try:
                d.text(view="bogus")  # type: ignore[arg-type]
            except ValueError as exc:
                assert "Unknown view" in str(exc)
            else:
                raise AssertionError("expected ValueError")


def describe_mermaid_renderer():
    def when_view_choreography():
        def it_renders_a_graph_lr_flowchart():

            output = EventGraph([place]).domain().mermaid()
            assert "graph LR" in output
            assert "Place -->|place| Placed" in output

        def it_uses_thick_entry_edges_for_seed_events():

            output = EventGraph([place]).domain().mermaid()
            assert "classDef entry fill:none,stroke:none,color:none" in output
            assert "==> Place" in output

    def when_outcome_has_no_direct_flow_edge():
        def it_emits_a_dashed_ownership_arrow_to_the_outcome():
            # Order.Place has two outcomes (Placed, Rejected). `place`
            # produces Placed; Rejected is only produced by the
            # `explain_rejection` policy reacting to InvariantViolated.
            # So Place→Rejected has no direct flow edge — ownership fills
            # the gap.
            output = EventGraph([place, explain_rejection_fixture]).domain().mermaid()
            assert "Place -.- Rejected" in output
            # Direct flow pair Place→Placed stays a solid arrow, NOT an
            # ownership arrow.
            assert "Place -.- Placed" not in output
            assert "stroke:#9ca3af" in output

    def when_invariant_has_pinned_reactor():
        # inv_place declares CustomerExists, inv_explain is pinned
        # (@on(InvariantViolated, invariant=CustomerExists)).

        def it_routes_reactor_edge_through_the_invariant():
            # The reactor's output edge leaves the Invariant diamond,
            # not InvariantViolated: Place -> Invariant -> (reactor target).
            @on(InvariantViolated, invariant=CustomerExists)
            def reject_pinned(event: InvariantViolated) -> Order.Place.Rejected:
                return Order.Place.Rejected(reason="banned")

            output = EventGraph([inv_place, reject_pinned]).domain().mermaid()
            assert "CustomerExists -.->|reject_pinned| Rejected" in output
            # No direct InvariantViolated -> Rejected edge.
            assert "InvariantViolated -->|reject_pinned|" not in output
            assert "InvariantViolated -.->|reject_pinned|" not in output

        def when_no_catchall_exists():

            def it_hides_the_InvariantViolated_node():
                @on(InvariantViolated, invariant=CustomerExists)
                def reject_pinned(event: InvariantViolated) -> Order.Place.Rejected:
                    return Order.Place.Rejected(reason="banned")

                output = EventGraph([inv_place, reject_pinned]).domain().mermaid()
                # InvariantViolated system-event node drops out entirely.
                assert "InvariantViolated([InvariantViolated]):::syst" not in output

            def it_drops_the_ownership_arrow_for_invariant_reached_outcomes():
                # Without the chain, Place -.- Rejected would appear
                # (ownership gap).  With the chain, the gap is covered.
                @on(InvariantViolated, invariant=CustomerExists)
                def reject_pinned(event: InvariantViolated) -> Order.Place.Rejected:
                    return Order.Place.Rejected(reason="banned")

                output = EventGraph([inv_place, reject_pinned]).domain().mermaid()
                assert "Place -.- Rejected" not in output

    def when_catchall_reactor_coexists():

        def it_keeps_the_InvariantViolated_node_for_the_catchall():
            # Catch-all reactor (@on(InvariantViolated) with no invariant=
            # pin) still routes through the InvariantViolated node; pinned
            # reactor routes through the Invariant diamond.
            @on(InvariantViolated, invariant=CustomerExists)
            def reject_pinned(event: InvariantViolated) -> Order.Place.Rejected:
                return Order.Place.Rejected(reason="pinned")

            @on(InvariantViolated)
            def audit_all(event: InvariantViolated) -> None:
                return None

            output = (
                EventGraph([inv_place, reject_pinned, audit_all]).domain().mermaid()
            )
            # Pinned reactor rerouted through Invariant.
            assert "CustomerExists -.->|reject_pinned| Rejected" in output
            # Catch-all keeps InvariantViolated visible (as side-effect
            # handler comment, since audit_all returns None).
            assert "InvariantViolated" in output

    def when_view_choreography_semantic_vocabulary():
        def it_declares_classdef_entries_for_each_taxonomy_class():
            output = _full_taxonomy_graph().domain().mermaid()
            assert "classDef cmd" in output
            assert "classDef devt" in output
            assert "classDef intg" in output
            assert "classDef syst" in output
            assert "classDef halt" in output

        def it_shapes_commands_as_hexes_and_domain_events_as_rounded():
            output = _full_taxonomy_graph().domain().mermaid()
            assert "Place{{Place}}:::cmd" in output
            assert "Placed(Placed):::devt" in output

        def it_wraps_domain_members_in_a_subgraph():
            output = _full_taxonomy_graph().domain().mermaid()
            assert 'subgraph Order["Order domain"]' in output
            assert "end" in output

        def when_halted_subtype_is_nested_in_domain():
            def it_places_it_inside_the_domain_subgraph():
                @on(_Content.Classify)
                def classify(
                    event: _Content.Classify,
                ) -> _Content.Classify.Classified | _Content.Blocked:
                    return _Content.Blocked(reason="test")

                output = EventGraph([classify]).domain().mermaid()
                subgraph_start = output.index('subgraph _Content["_Content domain"]')
                subgraph_end = output.index("end", subgraph_start)
                block = output[subgraph_start:subgraph_end]
                assert "Blocked([Blocked]):::halt" in block

        def when_handler_returns_typed_scatter():
            def it_emits_a_scatter_linkstyle_line():
                @on(_Batch)
                def split(event: _Batch) -> Scatter[_Item]:
                    return Scatter([_Item()])

                @on(_Item)
                def handle_item(event: _Item) -> None:
                    return None

                output = EventGraph([split, handle_item]).domain().mermaid()
                assert "linkStyle" in output
                assert "stroke:#7c3aed" in output

        def when_handler_declares_raises():
            def it_emits_a_raises_linkstyle_line():
                class _DemoError(Exception):
                    pass

                @on(Order.Place, raises=_DemoError)
                def place(event: Order.Place) -> Order.Place.Placed:
                    raise _DemoError

                @on(HandlerRaised, exception=_DemoError)
                def recover(event: HandlerRaised) -> None:
                    return None

                output = EventGraph([place, recover]).domain().mermaid()
                assert "stroke:#6b7280" in output


def describe_json_and_to_dict():
    def it_returns_json_serializable_dict():

        d = EventGraph([place]).domain()
        payload = d.to_dict()
        # Round-trip through JSON.
        reparsed = json.loads(json.dumps(payload))
        assert "Order" in reparsed["domains"]
        assert reparsed["domains"]["Order"]["commands"]["Place"]["type"].endswith(
            "Place"
        )
        assert any(ch["name"] == "place" for ch in reparsed["command_handlers"])

    def it_encodes_event_types_as_qualnames():

        d = EventGraph([place]).domain()
        payload = d.to_dict()
        place_cmd = payload["domains"]["Order"]["commands"]["Place"]
        assert place_cmd["type"] == Order.Place.__qualname__

    def it_encodes_policies_distinctly_from_command_handlers():

        payload = EventGraph([notify]).domain().to_dict()
        policy = next(p for p in payload["policies"] if p["name"] == "notify")
        assert policy["kind"] == "policy"
        assert Order.Shipped.__qualname__ in policy["subscribes"]

    def it_json_returns_string():

        d = EventGraph([place]).domain()
        js = d.json()
        assert isinstance(js, str)
        assert "domains" in js

    def it_encodes_invariants_block():
        payload = EventGraph([inv_place, inv_explain]).domain().to_dict()
        reparsed = json.loads(json.dumps(payload))
        encoded = next(
            i for i in reparsed["invariants"] if i["cls"].endswith("CustomerExists")
        )
        assert encoded["cls"] == CustomerExists.__qualname__
        assert Order.Place.__qualname__ in encoded["commands"]
        assert encoded["declared_by"] == ["inv_place"]
        assert encoded["reactors"] == ["inv_explain"]


# ---------------------------------------------------------------------------
# Tiny helper so the "unknown view" test can build a graph without requiring
# a real handler — an inline no-op command handler does the job.
# ---------------------------------------------------------------------------


def _placeholder_handler():
    @on(PaymentConfirmed)
    def _no_op(event: PaymentConfirmed) -> None:
        return None

    return _no_op
