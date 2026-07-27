"""Tests for QueryTool — the ReAct-loop tool over Reflection."""

from __future__ import annotations

import re

from conftest import Order

from langgraph_events import EventGraph


def _tool_and_reflection(seed=None):
    graph = EventGraph([Order.Place])
    seed = seed or Order.Place(customer_id="c1")
    reflection = graph.reflect(graph.invoke(seed))
    return reflection.tool(), reflection


def describe_tool():
    def when_inspecting_the_tool_shape():
        def it_is_named_query_log():
            tool, _ = _tool_and_reflection()

            assert tool.name == "query_log"

        def it_declares_op_type_index_and_limit_parameters():
            tool, _ = _tool_and_reflection()

            assert tool.parameters["type"] == "object"
            assert set(tool.parameters["properties"]) == {
                "op",
                "type",
                "index",
                "limit",
            }
            assert tool.parameters["required"] == ["op"]
            assert "enum" in tool.parameters["properties"]["op"]

        def it_embeds_the_event_type_vocabulary_in_the_description():
            tool, _ = _tool_and_reflection()

            assert "Order" in tool.description
            assert "Place" in tool.description
            assert "#" in tool.description  # teaches the #<index> convention

    def when_running_native_ops():
        def it_mirrors_filter_as_an_indexed_listing():
            tool, _ = _tool_and_reflection()

            out = tool.run(op="filter", type="Placed")

            assert "#1 Placed(order_id='o1')" in out

        def it_mirrors_count():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="count", type="Placed") == "1"

        def it_mirrors_has():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="has", type="Shipped") == "false"

        def it_mirrors_latest_as_a_single_line():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="latest", type="Placed").startswith("#1 Placed")

        def it_anchors_after_on_the_first_instance_preserving_root_indices():
            tool, _ = _tool_and_reflection()

            out = tool.run(op="after", type="Place")

            assert "#1 Placed" in out
            assert "#0" not in out

        def it_matches_base_kinds_subclass_aware():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="count", type="DomainEvent") == "1"

    def when_a_listing_exceeds_the_limit():
        def it_caps_and_reports_the_remainder():
            tool, _ = _tool_and_reflection(
                [Order.Place(customer_id=f"c{i}") for i in range(3)]
            )

            out = tool.run(op="filter", type="Place", limit=2)

            assert "and 1 more" in out

    def when_the_type_name_is_unknown():
        def it_returns_a_guidance_string():
            tool, _ = _tool_and_reflection()

            out = tool.run(op="filter", type="Placd")

            assert out.startswith("error: unknown type 'Placd'")
            assert "Placed" in out

    def when_the_index_is_out_of_range():
        def it_returns_a_range_guidance_string():
            tool, _ = _tool_and_reflection()

            out = tool.run(op="get", index=42)

            assert out.startswith("error: index 42 out of range")
            assert "0..1" in out

    def when_the_op_is_unknown():
        def it_returns_a_guidance_string_listing_valid_ops():
            tool, _ = _tool_and_reflection()

            out = tool.run(op="explain")

            assert out.startswith("error: unknown op 'explain'")
            assert "evidence" in out

    def when_running_reflection_ops():
        def it_delegates_overview_to_the_reflection_method():
            tool, reflection = _tool_and_reflection()

            assert tool.run(op="overview") == reflection.overview()

        def it_delegates_evidence():
            tool, reflection = _tool_and_reflection()

            assert tool.run(op="evidence", index=1) == reflection.evidence(1)

        def it_renders_state_as_text():
            tool, _ = _tool_and_reflection()

            assert "current_status: 'placed'" in tool.run(op="state")

    def when_driven_across_every_op():
        def it_returns_a_string_from_every_op():
            tool, _ = _tool_and_reflection()

            for op in tool.parameters["properties"]["op"]["enum"]:
                kwargs = {"op": op}
                if op in {
                    "filter",
                    "select",
                    "latest",
                    "first",
                    "has",
                    "count",
                    "after",
                    "before",
                }:
                    kwargs["type"] = "Place"
                if op in {"get", "evidence"}:
                    kwargs["index"] = 0

                assert isinstance(tool.run(**kwargs), str), op

        def it_supports_index_drilldown_round_trips():
            tool, _ = _tool_and_reflection()

            listing = tool.run(op="filter", type="Placed")
            index = int(re.search(r"#(\d+)", listing).group(1))

            assert tool.run(op="evidence", index=index).startswith("evidence for #1")
