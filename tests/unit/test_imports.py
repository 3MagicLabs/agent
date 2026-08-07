"""Import safety.

Two production outages came from module-level side effects: a search client
constructed at import time, and a chat model built at import time. Both crashed
the app before it could serve a single request. Nothing may import-fail without
credentials.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

MODULES = [
    "agent",
    "agent.cli",
    "agent.config",
    "agent.core",
    "agent.core.graph",
    "agent.core.llm",
    "agent.agents",
    "agent.agents.base",
    "agent.eval",
    "agent.eval.harness",
    "agent.obs",
    "agent.tools",
    "agent.tools.code",
    "agent.tools.files",
    "agent.tools.web",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_credentials(module):
    importlib.import_module(module)


def test_app_entry_point_imports_without_credentials():
    """The Hugging Face Space imports app.py before any request arrives."""
    importlib.import_module("app")


def test_public_api_is_stable():
    import agent

    for name in ("answer_question", "Orchestrator", "Settings", "get_settings", "__version__"):
        assert hasattr(agent, name), f"agent.{name} disappeared from the public API"


def test_cli_parser_builds():
    from agent.cli import build_parser

    args = build_parser().parse_args(["run", "--limit", "3"])

    assert args.limit == 3


class TestSpaceEntryPoint:
    """gradio's mocked OAuth validates the local HF token while Blocks closes,
    which raises during import. These pin the guard that avoids that path."""

    def test_imports_with_a_stale_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_invalid_token")
        monkeypatch.delenv("SPACE_ID", raising=False)
        importlib.reload(importlib.import_module("app"))

    def test_no_oauth_component_outside_a_space(self, monkeypatch):
        import app

        monkeypatch.delenv("SPACE_ID", raising=False)
        monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
        assert app.in_hosted_space() is False

    def test_oauth_only_when_hf_provisions_it(self, monkeypatch):
        import app

        monkeypatch.setenv("SPACE_ID", "user/space")
        monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
        assert app.in_hosted_space() is False, "SPACE_ID alone means mocked OAuth"

        monkeypatch.setenv("OAUTH_CLIENT_ID", "abc123")
        assert app.in_hosted_space() is True
