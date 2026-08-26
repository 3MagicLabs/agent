"""Conversation shaping: the rules that keep a message list provider-portable.

Each case here is a failure that was observed against a live provider, not a
hypothetical.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.core.conversation import (
    CONTINUE,
    ends_with_request,
    merge_system,
    normalize,
    text_of,
)

pytestmark = pytest.mark.unit


class TestTextOf:
    def test_a_plain_string_is_returned_as_is(self):
        assert text_of(AIMessage(content="FunkMonk")) == "FunkMonk"

    def test_text_blocks_are_joined(self):
        message = AIMessage(
            content=[{"type": "text", "text": "Funk"}, {"type": "text", "text": "Monk"}]
        )

        assert text_of(message) == "FunkMonk"

    def test_thinking_blocks_are_dropped(self):
        """A serialized thinking block once became the final answer."""
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "let me consider", "signature": "EsEECpAB"},
                {"type": "text", "text": "FunkMonk"},
            ]
        )

        assert text_of(message) == "FunkMonk"
        assert "EsEECpAB" not in text_of(message)

    def test_a_tool_use_block_contributes_nothing(self):
        message = AIMessage(
            content=[{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}]
        )

        assert text_of(message) == ""


class TestMergeSystem:
    def test_a_single_leading_system_message_is_untouched(self):
        messages = [SystemMessage(content="rules"), HumanMessage(content="q")]

        assert [type(m) for m in merge_system(messages)] == [SystemMessage, HumanMessage]

    def test_a_trailing_system_message_is_folded_into_the_leading_one(self):
        """Prepend-a-prompt then append-a-note is what the retry path did."""
        messages = [
            SystemMessage(content="rules"),
            HumanMessage(content="q"),
            SystemMessage(content="previous error: boom"),
        ]

        merged = merge_system(messages)

        assert [type(m) for m in merged] == [SystemMessage, HumanMessage]
        assert "rules" in str(merged[0].content)
        assert "previous error: boom" in str(merged[0].content)

    def test_the_order_of_other_messages_survives(self):
        messages = [
            HumanMessage(content="first"),
            SystemMessage(content="rules"),
            AIMessage(content="second"),
        ]

        merged = merge_system(messages)

        assert [text_of(m) for m in merged[1:]] == ["first", "second"]

    def test_no_system_message_means_no_change(self):
        messages = [HumanMessage(content="q"), AIMessage(content="a")]

        assert merge_system(messages) == messages

    def test_empty_system_content_does_not_add_blank_lines(self):
        messages = [SystemMessage(content="rules"), SystemMessage(content="   ")]

        assert str(merge_system(messages)[0].content) == "rules"


class TestEndsWithRequest:
    def test_a_trailing_assistant_turn_gets_a_request(self):
        """Anthropic reads a trailing assistant turn as a prefill and 400s."""
        messages = [HumanMessage(content="q"), AIMessage(content="[web_agent] found it")]

        ended = ends_with_request(messages)

        assert isinstance(ended[-1], HumanMessage)
        assert str(ended[-1].content) == CONTINUE

    def test_a_trailing_human_turn_is_left_alone(self):
        messages = [AIMessage(content="a"), HumanMessage(content="now answer")]

        assert ends_with_request(messages) == messages

    def test_a_tool_result_is_a_valid_ending(self):
        """A tool result maps to a user turn on the wire."""
        messages = [HumanMessage(content="q"), ToolMessage(content="42", tool_call_id="t1")]

        assert ends_with_request(messages) == messages

    def test_the_request_text_is_caller_supplied(self):
        messages = [AIMessage(content="a")]

        ended = ends_with_request(messages, "Choose the next specialist.")

        assert str(ended[-1].content) == "Choose the next specialist."

    def test_an_empty_conversation_still_gets_a_turn(self):
        assert len(ends_with_request([])) == 1


class TestNormalize:
    def test_both_rules_apply_together(self):
        """The exact shape that produced two different 400s on one task."""
        messages = [
            SystemMessage(content="rules"),
            HumanMessage(content="q"),
            AIMessage(content="[web_agent] partial"),
            SystemMessage(content="files already downloaded: a.png"),
        ]

        shaped = normalize(messages)

        assert [type(m) for m in shaped] == [
            SystemMessage,
            HumanMessage,
            AIMessage,
            HumanMessage,
        ]
        assert "rules" in str(shaped[0].content)
        assert "a.png" in str(shaped[0].content)

    def test_an_already_valid_conversation_is_unchanged(self):
        messages = [SystemMessage(content="rules"), HumanMessage(content="q")]

        assert normalize(messages) == messages
