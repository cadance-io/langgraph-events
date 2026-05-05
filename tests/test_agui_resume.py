"""Tests for AG-UI ResumeFactory helpers in `langgraph_events.agui`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from ag_ui.core.types import (
    ActivityMessage,
    AssistantMessage,
    BinaryInputContent,
    DeveloperMessage,
    FunctionCall,
    ReasoningMessage,
    RunAgentInput,
    TextInputContent,
    ToolCall,
    UserMessage,
)
from ag_ui.core.types import SystemMessage as AGUISystemMessage
from ag_ui.core.types import ToolMessage as AGUIToolMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_events.agui import (
    agui_messages_to_langchain,
    extract_resume_input,
    merge_frontend_messages,
)


def make_run_agent_input(
    *,
    messages: list[Any] | None = None,
    forwarded_props: Any = None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=messages or [],
        tools=[],
        context=[],
        forwarded_props=forwarded_props if forwarded_props is not None else {},
    )


def describe_agui_messages_to_langchain():
    def when_message_is_user():
        def when_content_is_string():
            def it_returns_a_human_message():
                msg = UserMessage(id="u1", content="hello")
                result = agui_messages_to_langchain([msg])
                assert len(result) == 1
                assert isinstance(result[0], HumanMessage)

            def it_preserves_id():
                msg = UserMessage(id="u1", content="hello")
                assert agui_messages_to_langchain([msg])[0].id == "u1"

            def it_preserves_content():
                msg = UserMessage(id="u1", content="hello")
                assert agui_messages_to_langchain([msg])[0].content == "hello"

            def it_preserves_name():
                msg = UserMessage(id="u1", content="hello", name="alice")
                assert agui_messages_to_langchain([msg])[0].name == "alice"

        def when_content_is_a_multimodal_list():
            def when_part_is_text():
                def it_emits_a_text_part_dict():
                    msg = UserMessage(
                        id="u1",
                        content=[TextInputContent(text="hi there")],
                    )
                    [out] = agui_messages_to_langchain([msg])
                    assert out.content == [{"type": "text", "text": "hi there"}]

            def when_part_is_binary():
                def with_url():
                    def it_uses_url_as_image_url():
                        msg = UserMessage(
                            id="u1",
                            content=[
                                BinaryInputContent(
                                    mime_type="image/png", url="https://x/y.png"
                                )
                            ],
                        )
                        [out] = agui_messages_to_langchain([msg])
                        assert out.content == [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://x/y.png"},
                            }
                        ]

                def with_data_only():
                    def it_builds_data_uri_using_mime_type():
                        msg = UserMessage(
                            id="u1",
                            content=[
                                BinaryInputContent(mime_type="image/png", data="AAA=")
                            ],
                        )
                        [out] = agui_messages_to_langchain([msg])
                        assert out.content == [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAA="},
                            }
                        ]

                def with_id_only():
                    def it_uses_id_as_image_url():
                        msg = UserMessage(
                            id="u1",
                            content=[
                                BinaryInputContent(mime_type="image/png", id="ref-42")
                            ],
                        )
                        [out] = agui_messages_to_langchain([msg])
                        assert out.content == [
                            {
                                "type": "image_url",
                                "image_url": {"url": "ref-42"},
                            }
                        ]

                def with_url_and_data():
                    def it_prefers_url_over_data():
                        msg = UserMessage(
                            id="u1",
                            content=[
                                BinaryInputContent(
                                    mime_type="image/png",
                                    url="https://x/y.png",
                                    data="AAA=",
                                )
                            ],
                        )
                        [out] = agui_messages_to_langchain([msg])
                        assert out.content[0]["image_url"]["url"] == "https://x/y.png"

    def when_message_is_assistant():
        def when_content_is_none_and_no_tool_calls():
            def it_returns_an_ai_message():
                msg = AssistantMessage(id="a1", content=None)
                [out] = agui_messages_to_langchain([msg])
                assert isinstance(out, AIMessage)

            def it_emits_empty_string_content():
                msg = AssistantMessage(id="a1", content=None)
                [out] = agui_messages_to_langchain([msg])
                assert out.content == ""

        def when_content_is_a_string_and_no_tool_calls():
            def it_passes_content_through():
                msg = AssistantMessage(id="a1", content="reply")
                [out] = agui_messages_to_langchain([msg])
                assert out.content == "reply"

            def it_preserves_id_and_name():
                msg = AssistantMessage(id="a1", content="r", name="bot")
                [out] = agui_messages_to_langchain([msg])
                assert out.id == "a1"
                assert out.name == "bot"

        def when_message_has_tool_calls():
            def it_parses_arguments_json_into_args_dict():
                msg = AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            function=FunctionCall(
                                name="search",
                                arguments='{"q": "hello"}',
                            ),
                        )
                    ],
                )
                [out] = agui_messages_to_langchain([msg])
                assert out.tool_calls == [
                    {
                        "id": "tc1",
                        "name": "search",
                        "args": {"q": "hello"},
                        "type": "tool_call",
                    }
                ]

            def when_arguments_string_is_empty():
                def it_defaults_args_to_empty_dict():
                    msg = AssistantMessage(
                        id="a1",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="tc1",
                                function=FunctionCall(name="ping", arguments=""),
                            )
                        ],
                    )
                    [out] = agui_messages_to_langchain([msg])
                    assert out.tool_calls[0]["args"] == {}

    def when_message_is_system():
        def it_returns_a_langchain_system_message():
            msg = AGUISystemMessage(id="s1", content="be helpful")
            [out] = agui_messages_to_langchain([msg])
            assert isinstance(out, SystemMessage)
            assert out.id == "s1"
            assert out.content == "be helpful"

        def it_preserves_name():
            msg = AGUISystemMessage(id="s1", content="x", name="root")
            [out] = agui_messages_to_langchain([msg])
            assert out.name == "root"

    def when_message_is_tool():
        def it_returns_a_langchain_tool_message():
            msg = AGUIToolMessage(id="tm1", content="42", tool_call_id="tc1")
            [out] = agui_messages_to_langchain([msg])
            assert isinstance(out, ToolMessage)
            assert out.id == "tm1"
            assert out.content == "42"
            assert out.tool_call_id == "tc1"

        def it_does_not_propagate_error_field():
            msg = AGUIToolMessage(
                id="tm1", content="42", tool_call_id="tc1", error="boom"
            )
            [out] = agui_messages_to_langchain([msg])
            assert getattr(out, "status", None) != "error"

    def when_role_is_reasoning():
        def it_skips_silently():
            msg = ReasoningMessage(id="r1", content="thinking")
            assert agui_messages_to_langchain([msg]) == []

        def it_logs_at_debug_naming_the_role(caplog):
            import logging

            caplog.set_level(logging.DEBUG, logger="langgraph_events.agui._resume")
            msg = ReasoningMessage(id="r1", content="thinking")
            agui_messages_to_langchain([msg])
            assert any("reasoning" in r.message for r in caplog.records)

    def when_role_is_developer():
        def it_skips_silently():
            msg = DeveloperMessage(id="d1", content="hint")
            assert agui_messages_to_langchain([msg]) == []

        def it_logs_at_debug_naming_the_role(caplog):
            import logging

            caplog.set_level(logging.DEBUG, logger="langgraph_events.agui._resume")
            msg = DeveloperMessage(id="d1", content="hint")
            agui_messages_to_langchain([msg])
            assert any("developer" in r.message for r in caplog.records)

    def when_role_is_activity():
        def it_raises_value_error_naming_the_role():
            msg = ActivityMessage(id="x1", activity_type="typing", content={})
            with pytest.raises(ValueError, match="activity"):
                agui_messages_to_langchain([msg])

    def when_role_is_unknown():
        def it_raises_value_error_naming_the_role():
            class Foreign:
                role = "alien"
                id = "z1"

            with pytest.raises(ValueError, match="alien"):
                agui_messages_to_langchain([Foreign()])  # type: ignore[list-item]

    def when_role_attribute_is_missing():
        def it_raises_value_error_naming_the_class():
            class NoRole:
                id = "z1"

            with pytest.raises(ValueError, match="NoRole"):
                agui_messages_to_langchain([NoRole()])  # type: ignore[list-item]

    def when_message_list_is_empty():
        def it_returns_empty_list():
            assert agui_messages_to_langchain([]) == []

    def when_messages_are_mixed():
        def it_preserves_order_skipping_reasoning_in_place():
            user = UserMessage(id="u1", content="hi")
            reasoning = ReasoningMessage(id="r1", content="...")
            assistant = AssistantMessage(id="a1", content="ok")
            out = agui_messages_to_langchain([user, reasoning, assistant])
            assert [m.id for m in out] == ["u1", "a1"]
            assert isinstance(out[0], HumanMessage)
            assert isinstance(out[1], AIMessage)

    def when_drop_invalid_tool_calls_is_false_by_default():
        def when_arguments_are_malformed():
            def it_propagates_json_decode_error():
                msg = AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            function=FunctionCall(name="f", arguments="not json"),
                        )
                    ],
                )
                with pytest.raises(json.JSONDecodeError):
                    agui_messages_to_langchain([msg])

    def when_drop_invalid_tool_calls_is_true():
        def when_one_tool_call_has_bad_json():
            def it_drops_only_that_tool_call():
                msg = AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="bad",
                            function=FunctionCall(name="f", arguments="not json"),
                        ),
                        ToolCall(
                            id="good",
                            function=FunctionCall(name="g", arguments='{"x": 1}'),
                        ),
                    ],
                )
                [out] = agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                assert [tc["id"] for tc in out.tool_calls] == ["good"]

            def it_logs_warning_naming_the_tool_call(caplog):
                msg = AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="bad",
                            function=FunctionCall(name="f", arguments="not json"),
                        ),
                        ToolCall(
                            id="good",
                            function=FunctionCall(name="g", arguments='{"x": 1}'),
                        ),
                    ],
                )
                agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                assert any("bad" in r.message for r in caplog.records)

        def when_all_tool_calls_have_bad_json():
            def it_drops_the_assistant_message():
                msg = AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="t",
                            function=FunctionCall(name="f", arguments="bad"),
                        )
                    ],
                )
                assert (
                    agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                    == []
                )

            def it_logs_warning_naming_the_message(caplog):
                msg = AssistantMessage(
                    id="m99",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="t",
                            function=FunctionCall(name="f", arguments="bad"),
                        )
                    ],
                )
                agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                assert any("m99" in r.message for r in caplog.records)

        def when_message_has_no_tool_calls():
            def it_passes_through_unchanged():
                msg = AssistantMessage(id="a1", content="hello")
                [out] = agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                assert out.content == "hello"

        def when_tool_calls_is_an_empty_list():
            def it_passes_the_message_through():
                msg = AssistantMessage(id="a1", content="hi", tool_calls=[])
                [out] = agui_messages_to_langchain([msg], drop_invalid_tool_calls=True)
                assert out.content == "hi"
                assert out.tool_calls == []


def describe_extract_resume_input():
    def when_forwarded_props_is_empty():
        def it_returns_none():
            assert extract_resume_input(make_run_agent_input()) is None

    def when_command_key_is_absent():
        def it_returns_none():
            inp = make_run_agent_input(forwarded_props={"other": 1})
            assert extract_resume_input(inp) is None

    def when_resume_key_is_absent():
        def it_returns_none():
            inp = make_run_agent_input(forwarded_props={"command": {}})
            assert extract_resume_input(inp) is None

    def when_resume_is_empty_string():
        def it_returns_none():
            inp = make_run_agent_input(forwarded_props={"command": {"resume": ""}})
            assert extract_resume_input(inp) is None

    def when_resume_is_a_dict():
        def it_returns_dict_unchanged():
            inp = make_run_agent_input(
                forwarded_props={"command": {"resume": {"k": "v"}}}
            )
            assert extract_resume_input(inp) == {"k": "v"}

    def when_resume_is_a_json_string():
        def it_decodes_to_its_json_value():
            inp = make_run_agent_input(
                forwarded_props={"command": {"resume": '{"k": 2}'}}
            )
            assert extract_resume_input(inp) == {"k": 2}

    def when_resume_is_a_non_json_string():
        def it_returns_string_as_is():
            inp = make_run_agent_input(
                forwarded_props={"command": {"resume": "approve"}}
            )
            assert extract_resume_input(inp) == "approve"

    def when_resume_is_a_list():
        def it_returns_list_unchanged():
            inp = make_run_agent_input(forwarded_props={"command": {"resume": [1, 2]}})
            assert extract_resume_input(inp) == [1, 2]

    def when_resume_is_a_nonzero_number():
        def it_returns_number_unchanged():
            inp = make_run_agent_input(forwarded_props={"command": {"resume": 7}})
            assert extract_resume_input(inp) == 7

    def when_resume_is_zero():
        def it_returns_none():
            inp = make_run_agent_input(forwarded_props={"command": {"resume": 0}})
            assert extract_resume_input(inp) is None


def describe_merge_frontend_messages():
    def when_checkpoint_state_is_none():
        def it_returns_only_converted_input_messages():
            inp = make_run_agent_input(messages=[UserMessage(id="u1", content="hi")])
            result = merge_frontend_messages(inp, None)
            assert len(result) == 1
            assert isinstance(result[0], HumanMessage)
            assert result[0].id == "u1"

        def it_returns_a_tuple():
            inp = make_run_agent_input(messages=[UserMessage(id="u1", content="hi")])
            assert isinstance(merge_frontend_messages(inp, None), tuple)

    def when_checkpoint_has_no_reducers_key():
        def it_returns_only_converted_input_messages():
            inp = make_run_agent_input(messages=[UserMessage(id="u1", content="hi")])
            result = merge_frontend_messages(inp, {})
            assert len(result) == 1
            assert result[0].id == "u1"

    def when_checkpoint_has_reducers_but_no_messages_reducer():
        def it_returns_only_converted_input_messages():
            inp = make_run_agent_input(messages=[UserMessage(id="u1", content="hi")])
            result = merge_frontend_messages(inp, {"reducers": {}})
            assert len(result) == 1
            assert result[0].id == "u1"

    def when_existing_and_new_share_id():
        def it_dedups_via_add_messages():
            existing = [HumanMessage(id="u1", content="old")]
            inp = make_run_agent_input(messages=[UserMessage(id="u1", content="new")])
            result = merge_frontend_messages(inp, {"reducers": {"messages": existing}})
            assert len(result) == 1
            assert result[0].content == "new"

    def when_input_messages_are_empty():
        def it_returns_existing_messages_unchanged():
            existing = [HumanMessage(id="u1", content="kept")]
            inp = make_run_agent_input()
            result = merge_frontend_messages(inp, {"reducers": {"messages": existing}})
            assert len(result) == 1
            assert result[0].id == "u1"
            assert result[0].content == "kept"

    def when_reducer_name_is_overridden():
        def it_pulls_from_the_named_reducer():
            existing = [HumanMessage(id="u1", content="kept")]
            inp = make_run_agent_input(messages=[UserMessage(id="u2", content="new")])
            result = merge_frontend_messages(
                inp,
                {"reducers": {"chat": existing}},
                reducer_name="chat",
            )
            assert [m.id for m in result] == ["u1", "u2"]

    def when_drop_invalid_tool_calls_is_default():
        def it_drops_invalid_tool_calls_silently():
            msg = AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="bad",
                        function=FunctionCall(name="x", arguments="not-json"),
                    )
                ],
            )
            inp = make_run_agent_input(messages=[msg])
            result = merge_frontend_messages(inp, None)
            assert result == ()

    def when_drop_invalid_tool_calls_is_false():
        def it_propagates_json_decode_error():
            msg = AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="bad",
                        function=FunctionCall(name="x", arguments="not-json"),
                    )
                ],
            )
            inp = make_run_agent_input(messages=[msg])
            with pytest.raises(json.JSONDecodeError):
                merge_frontend_messages(inp, None, drop_invalid_tool_calls=False)
