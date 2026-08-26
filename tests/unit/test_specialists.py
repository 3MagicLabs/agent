"""The ReAct loop in ``agents.base``: retries, budgets and output isolation."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.agents.base import SpecialistSpec, build_specialist, tool_evidence
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
    """A permanently broken provider must not loop until the recursion limit."""
    llm = FlakyLLM(failures=99)

    build_specialist(make_spec(max_iterations=2), llm_factory=lambda: llm).invoke(initial())

    assert len(llm.calls) == 2


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
