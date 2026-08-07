"""Integration tests against live providers.

Skipped unless credentials are present, so the default suite stays offline.
Run with: pytest -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

needs_llm = pytest.mark.skipif(
    not any(os.getenv(k) for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")),
    reason="no LLM provider credentials configured",
)
needs_network = pytest.mark.skipif(os.getenv("OFFLINE") == "1", reason="OFFLINE=1 set")


@pytest.fixture(autouse=True)
def live_env(monkeypatch, tmp_path):
    """Use the real environment, but keep artefacts in a temp directory."""
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    from agent.config import reset_settings

    reset_settings()
    yield
    reset_settings()


@needs_llm
def test_answers_a_trivial_question() -> None:
    from agent.core.graph import Orchestrator

    answer = Orchestrator().answer("What is 2 + 2? Reply with the number only.", task_id="live-1")

    assert "4" in answer


@needs_llm
def test_router_produces_a_valid_choice() -> None:
    """Small models often fail structured output; this catches it early."""
    from agent.core.graph import Orchestrator

    orchestrator = Orchestrator()
    result = orchestrator._supervise({"messages": [], "steps": 0})

    valid = {spec.name for spec in orchestrator.specs} | {"FINISH"}
    assert result["next_agent"] in valid


@needs_network
def test_scoring_api_is_reachable() -> None:
    from agent.eval import BenchmarkRunner

    questions = BenchmarkRunner().fetch_questions()

    assert questions
    assert "task_id" in questions[0]


@needs_network
def test_wikipedia_tool_returns_content() -> None:
    from agent.tools.web import wikipedia_lookup

    result = wikipedia_lookup.invoke({"title": "Mars"})

    assert "Mars" in result
    assert len(result) > 200
