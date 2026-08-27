"""Shared fixtures.

Every unit test runs with a clean environment and a stub model, so the suite
needs no credentials and makes no network calls.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `import app` (the Space entry point) resolves

from agent.config import Settings, reset_settings, set_settings

CREDENTIAL_VARS = (
    "LLM_PROVIDER",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "TAVILY_API_KEY",
    "E2B_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
)


class StubRouter:
    """Stands in for ``llm.with_structured_output(..., include_raw=True)``.

    Returns the {"raw", "parsed", "parsing_error"} envelope rather than the
    parsed object. The graph reads ``raw`` to tell a policy refusal - a 200
    with an empty body and stop_reason "refusal" - from a merely malformed
    reply, because one is worth retrying and the other never is.

    ``refusal`` makes the stub decline, so that branch is testable offline.
    """

    def __init__(
        self,
        model_cls: type,
        next_agent: str,
        reasoning: str = "stub",
        refusal: str = "",
    ) -> None:
        self._model_cls = model_cls
        self._next_agent = next_agent
        self._reasoning = reasoning
        self._refusal = refusal
        self.calls = 0

    def invoke(self, _messages: Sequence[BaseMessage]) -> Any:
        self.calls += 1
        if self._refusal:
            declined = AIMessage(
                content="",
                response_metadata={
                    "stop_reason": "refusal",
                    "stop_details": {"type": "refusal", "category": self._refusal},
                },
            )
            return {"raw": declined, "parsed": None, "parsing_error": None}
        parsed = self._model_cls(next_agent=self._next_agent, reasoning=self._reasoning)
        return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}


class StubLLM:
    """Deterministic chat model. ``route_to`` drives the supervisor's choice."""

    def __init__(
        self, reply: str = "stub answer", route_to: str = "FINISH", refusal: str = ""
    ) -> None:
        self.reply = reply
        self.route_to = route_to
        self.refusal = refusal
        self.router: StubRouter | None = None
        self.calls: list[list[BaseMessage]] = []
        self.bound: dict[str, Any] = {}

    def with_structured_output(self, model_cls: type, **_kwargs: Any) -> StubRouter:
        self.router = StubRouter(model_cls, self.route_to, refusal=self.refusal)
        return self.router

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> StubLLM:
        return self

    def bind(self, **kwargs: Any) -> StubLLM:
        """Record per-call overrides (``max_tokens``) and stay chainable."""
        self.bound = {**self.bound, **kwargs}
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


class FailingLLM:
    """Raises on every call, to exercise error paths."""

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> FailingLLM:
        return self

    def bind_tools(self, *_args: Any, **_kwargs: Any) -> FailingLLM:
        return self

    def bind(self, **_kwargs: Any) -> FailingLLM:
        return self

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider exploded")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No credentials, and all writes land in a temp directory."""
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    set_settings(Settings(log_dir=tmp_path / "logs", download_dir=tmp_path / "downloads"))

    from agent.tools.files import _INDEX

    _INDEX.clear()  # a listing cached under other settings must not leak
    yield
    reset_settings()


@pytest.fixture
def settings():
    from agent.config import get_settings

    return get_settings()


@pytest.fixture
def stub_llm(monkeypatch):
    """Install a StubLLM everywhere the graph resolves a model."""

    def _install(
        reply: str = "stub answer", route_to: str = "FINISH", refusal: str = ""
    ) -> StubLLM:
        llm = StubLLM(reply=reply, route_to=route_to, refusal=refusal)
        monkeypatch.setattr("agent.core.graph.get_llm", lambda: llm)
        monkeypatch.setattr("agent.agents.base.get_llm", lambda: llm)
        return llm

    return _install


@pytest.fixture
def failing_llm(monkeypatch):
    def _install() -> FailingLLM:
        llm = FailingLLM()
        monkeypatch.setattr("agent.core.graph.get_llm", lambda: llm)
        monkeypatch.setattr("agent.agents.base.get_llm", lambda: llm)
        return llm

    return _install
