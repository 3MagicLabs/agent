"""Supervisor termination and error containment.

The original design had no step counter and no recursion limit, so a router
that never said FINISH looped until LangGraph aborted. These tests pin that
shut.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.core.graph import Orchestrator, build_route_model, routing_prompt, trim

pytestmark = pytest.mark.unit


def test_router_choices_match_live_specialists(settings):
    orchestrator = Orchestrator(settings)
    model = build_route_model(orchestrator.specs)
    choices = model.model_fields["next_agent"].annotation.__args__

    assert set(choices) == {"web_agent", "code_agent", "FINISH"}


def test_routing_prompt_lists_every_specialist(settings):
    orchestrator = Orchestrator(settings)
    prompt = routing_prompt(orchestrator.specs)

    for spec in orchestrator.specs:
        assert spec.name in prompt
        assert spec.description in prompt


def test_terminates_when_router_never_finishes(settings, stub_llm):
    """The pathological case: always delegate, never FINISH."""
    stub_llm(reply="web result", route_to="web_agent")

    orchestrator = Orchestrator(settings)
    state = orchestrator.graph.invoke(
        {"messages": [HumanMessage(content="how many moons does Mars have?")], "steps": 0},
        config={"recursion_limit": settings.recursion_limit},
    )

    assert state["steps"] <= settings.max_supervisor_steps + 1
    assert state["messages"][-1].name == "final_answer"


def test_finish_route_reaches_the_finalizer(settings, stub_llm):
    stub_llm(reply="2", route_to="FINISH")

    answer = Orchestrator(settings).answer("how many moons?", task_id="t1")

    assert answer == "2"


def test_unknown_route_falls_through_to_finalize(settings, stub_llm):
    """A hallucinated agent name must not raise a KeyError."""
    stub_llm(reply="fallback", route_to="FINISH")
    orchestrator = Orchestrator(settings)

    assert orchestrator._route({"next_agent": "nonexistent_agent"}) == "finalize"


def test_provider_failure_does_not_crash_the_run(settings, failing_llm):
    """Every LLM call raises; the run must still return a string."""
    failing_llm()

    answer = Orchestrator(settings).answer("anything", task_id="t2")

    assert isinstance(answer, str)


def test_answer_is_not_prefixed_with_the_specialist_label(settings, stub_llm):
    """Submitting '[web_agent] ...' scores zero on an exact-match benchmark."""
    stub_llm(reply="42", route_to="web_agent")

    answer = Orchestrator(settings).answer("six times seven?", task_id="t3")

    assert not answer.startswith("[web_agent]")


class TestTrim:
    def test_keeps_everything_below_the_window(self):
        messages = [HumanMessage(content=str(i)) for i in range(3)]
        assert trim(messages, keep=8) == messages

    def test_always_keeps_the_original_question(self):
        messages = [
            HumanMessage(content="original"),
            *[AIMessage(content=str(i)) for i in range(20)],
        ]

        trimmed = trim(messages, keep=4)

        assert len(trimmed) == 4
        assert trimmed[0].content == "original"
        assert trimmed[-1].content == "19"
