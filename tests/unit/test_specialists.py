"""The ReAct loop in ``agents.base``: retries, budgets and output isolation."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from tests.conftest import ToolCallingLLM

from agent.agents.base import (
    SpecialistSpec,
    build_specialist,
    last_text,
    tool_evidence,
)
from agent.tools import get_tools

pytestmark = pytest.mark.unit


class FlakyLLM:
    """Fails the first ``failures`` calls, then answers."""

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> FlakyLLM:
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if len(self.calls) <= self.failures:
            raise RuntimeError("tool_use_failed: malformed function call")
        return AIMessage(content="42")


class StubLLM:
    """Always answers, never calls tools."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> StubLLM:
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


def make_spec(max_iterations: int = 3) -> SpecialistSpec:
    return SpecialistSpec(
        name="probe",
        description="test specialist",
        prompt="you are a probe",
        tools=get_tools("files"),
        max_iterations=max_iterations,
    )


def initial(question: str = "how many?") -> dict[str, Any]:
    return {"messages": [HumanMessage(content=question)], "iterations": 0, "last_error": ""}


def test_a_failed_call_is_retried_with_the_error_quoted():
    """Groq rejects malformed tool calls with a 400; one retry usually fixes it.

    ``last_error`` used to be hardcoded to "" on every path, so the retry branch
    in ``reason`` was unreachable and a single bad call ended the specialist.
    """
    llm = FlakyLLM(failures=1)

    result = build_specialist(make_spec(), llm_factory=lambda: llm).invoke(initial())

    assert len(llm.calls) == 2
    assert "Previous error to fix" in str(llm.calls[1][-1].content)
    assert "42" in str(result["messages"][-1].content)


def test_retries_stop_at_the_iteration_cap():
    """A permanently broken provider must not loop until the recursion limit.

    Two reasoning turns, then one wrap-up. The wrap-up is bounded too - it has
    no tools and cannot route back - so a broken provider costs exactly
    max_iterations + 1 calls, not an unbounded number.
    """
    llm = FlakyLLM(failures=99)

    build_specialist(make_spec(max_iterations=2), llm_factory=lambda: llm).invoke(initial())

    assert len(llm.calls) == 3


def test_a_successful_call_clears_the_error():
    llm = FlakyLLM(failures=1)

    result = build_specialist(make_spec(), llm_factory=lambda: llm).invoke(initial())

    assert result["last_error"] == ""


class TestToolEvidence:
    """The supervisor sees only a specialist's text, so it needs provenance.

    Without it a researched answer and an invented one look identical, and the
    supervisor re-delegates to verify - four rounds and 34,185 tokens on a task
    that was solved in round one.
    """

    def test_a_tool_that_ran_is_named(self):
        messages = [ToolMessage(content="results", name="web_search", tool_call_id="1")]

        assert tool_evidence(messages) == "web_search"

    def test_repeats_are_counted(self):
        messages = [
            ToolMessage(content="a", name="web_search", tool_call_id="1"),
            ToolMessage(content="b", name="web_search", tool_call_id="2"),
        ]

        assert tool_evidence(messages) == "web_search x2"

    def test_several_tools_are_listed_in_a_stable_order(self):
        messages = [
            ToolMessage(content="a", name="web_search", tool_call_id="1"),
            ToolMessage(content="b", name="scrape_webpage", tool_call_id="2"),
        ]

        assert tool_evidence(messages) == "scrape_webpage, web_search"

    def test_no_tools_is_stated_explicitly(self):
        """Silence reads as 'maybe searched'; absence of evidence must be visible."""
        assert "unverified" in tool_evidence([AIMessage(content="FunkMonk")])

    def test_a_requested_but_unexecuted_call_is_not_evidence(self):
        """A tool_call can be emitted and never run; only a result proves it did."""
        requested = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}],
        )

        assert "unverified" in tool_evidence([requested])


class TestToollessEvidence:
    """A specialist with no tools has not failed to use them."""

    def test_a_toolless_specialist_is_not_marked_unverified(self):
        """reason_agent has tools=() by design, so 'unverified' made the
        supervisor re-delegate after every single reasoning turn."""
        evidence = tool_evidence([AIMessage(content="b, e")], has_tools=False)

        assert "unverified" not in evidence
        assert "no tools by design" in evidence

    def test_a_tooled_specialist_that_used_none_is_still_unverified(self):
        evidence = tool_evidence([AIMessage(content="probably 3")], has_tools=True)

        assert "unverified" in evidence

    def test_tools_that_ran_are_reported_either_way(self):
        messages = [ToolMessage(content="r", name="web_search", tool_call_id="1")]

        assert tool_evidence(messages, has_tools=True) == "web_search"
        assert tool_evidence(messages, has_tools=False) == "web_search"


class TestWrapUp:
    """Work done but never reported is work paid for twice.

    These drive a specialist with a stub that really emits tool calls and really
    enforces the provider's message rules. Asserting on the returned *text* is
    not enough: summarize catches every exception and substitutes "ran out of
    steps before reporting", so a contract violation reads exactly like an
    honest budget exhaustion - which is how the same 400 shipped twice.
    """

    def test_a_capped_specialist_still_reports(self):
        """Hitting the budget used to end the subgraph outright, so a specialist
        that had downloaded, read and computed had no turn left to say what it
        found - and the supervisor re-delegated the whole job."""
        llm = ToolCallingLLM(script=["read_file"], reply="the total is 89706.00")

        result = build_specialist(make_spec(max_iterations=1), llm_factory=lambda: llm).invoke(
            initial()
        )

        assert "89706.00" in last_text(list(result["messages"]))

    def test_the_wrap_up_conversation_is_well_formed(self):
        """The assertion that matters is on what was SENT, not what came back."""
        llm = ToolCallingLLM(script=["read_file"], reply="done")

        build_specialist(make_spec(max_iterations=1), llm_factory=lambda: llm).invoke(initial())

        final = llm.calls[-1]
        resolved = {m.tool_call_id for m in final if isinstance(m, ToolMessage)}
        for message in final:
            for call in getattr(message, "tool_calls", None) or []:
                assert str(call["id"]) in resolved, "an unrun tool call reached the provider"

    def test_the_wrap_up_did_not_silently_fail(self):
        """ "ran out of steps" is the fallback that hid two shipped 400s."""
        llm = ToolCallingLLM(script=["read_file"], reply="the total is 89706.00")

        result = build_specialist(make_spec(max_iterations=1), llm_factory=lambda: llm).invoke(
            initial()
        )

        assert "ran out of steps" not in last_text(list(result["messages"]))

    def test_the_wrap_up_turn_is_told_not_to_call_tools(self):
        llm = ToolCallingLLM(script=["read_file"], reply="done")

        build_specialist(make_spec(max_iterations=1), llm_factory=lambda: llm).invoke(initial())

        assert any("do not call any more tools" in str(c[-1].content).lower() for c in llm.calls)

    def test_a_failed_wrap_up_does_not_kill_the_run(self):
        llm = FlakyLLM(failures=99)

        result = build_specialist(make_spec(max_iterations=1), llm_factory=lambda: llm).invoke(
            initial()
        )

        assert "ran out of steps" in last_text(list(result["messages"]))


class TestBudgetCountsToolCalls:
    """The budget bounds tool calls, not thoughts.

    Counting every reasoning turn meant a tool call and the thought producing it
    each cost one, so six turns bought five tools and nothing to report with -
    the whole reason the summarize node had to be written.
    """

    def test_an_answer_without_tools_costs_nothing(self):
        llm = ToolCallingLLM(script=[], reply="42")

        result = build_specialist(make_spec(max_iterations=3), llm_factory=lambda: llm).invoke(
            initial()
        )

        assert result["iterations"] == 0
        assert len(llm.calls) == 1

    def test_a_finished_specialist_is_not_sent_to_wrap_up(self):
        """It answered on its last allowed turn; summarising replaces a good
        answer with a paraphrase of itself."""
        llm = ToolCallingLLM(script=["read_file"], reply="the answer")

        build_specialist(make_spec(max_iterations=2), llm_factory=lambda: llm).invoke(initial())

        assert not any(
            "do not call any more tools" in str(c[-1].content).lower() for c in llm.calls
        )
